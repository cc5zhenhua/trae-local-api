/**
 * OpenAI Responses API (POST /v1/responses) handler factory.
 *
 * Translates the Responses-style `input`/`instructions` request body into the
 * chat messages consumed by the existing llmUtilsChat streaming engine, then
 * emits Responses-shaped events (streaming) or a `response` object
 * (non-streaming). Reuses the same fallback chain as /v1/chat/completions.
 */
const path = require('path');
const fs = require('fs');
const { v4: uuidv4 } = require('./uuid');
const { createResponsesResponse } = require('./openai-format');
const trafficLogger = require('./traffic-logger');

module.exports = function createResponsesHandler(deps) {
  return makeHandler(deps);
};

function makeHandler(deps) {
  const {
    responsesInputToMessages,
    resolveModelId, MODEL_MAP, findMultimodalModel,
    refreshTokenIfNeeded, isTokenExpired,
    llmUtilsChat, chatCompletion, createAgentTask,
    parseLlmUtilsChatStream, parseAgentTaskStream, parseTraeStreamChunk,
    sessionsRepo, extractWorkspace, WORKSPACE_DIR, syncFileToOutput,
  } = deps;

  return async function responsesHandler(req, res) {
    const reqId = uuidv4().substring(0, 8);
    try {
      const body = req.body || {};
      const {
        input, model, instructions, stream,
        function: funcName, config_name, workspace_dir, save_to, previous_response_id,
      } = body;
      const tools = body.tools;
      const toolCount = Array.isArray(tools) ? tools.length : 0;
      const inputKind = input == null ? 'null' : Array.isArray(input) ? `array[${input.length}]` : typeof input;
      console.log(`[responses ${reqId}] POST /v1/responses ua=${req.headers['user-agent'] || '-'} keys=${Object.keys(body).join(',')}`);
      console.log(`[responses ${reqId}] model=${model || 'auto'} stream=${stream !== false} input=${inputKind} tools=${toolCount} instructions_len=${(instructions && String(instructions).length) || 0} prev=${previous_response_id || 'none'}`);
      if (input === undefined) {
        console.error(`[responses ${reqId}] REJECT 400: input is required`);
        return res.status(400).json({ error: { message: 'input is required', type: 'invalid_request_error' } });
      }
      const messages = responsesInputToMessages(input, instructions);
      if (messages.length === 0) {
        console.error(`[responses ${reqId}] REJECT 400: input resolved to no messages`);
        return res.status(400).json({ error: { message: 'input resolved to no messages', type: 'invalid_request_error' } });
      }
      const modelName = model || 'auto';
      const isStream = stream !== false;
      const respId = `resp_${uuidv4().replace(/-/g, '').substring(0, 24)}`;
      const msgId = `msg_${uuidv4().replace(/-/g, '').substring(0, 24)}`;
      const lastUser = [...messages].reverse().find(m => m && m.role === 'user');
      const preview = lastUser && lastUser.content != null
        ? String(typeof lastUser.content === 'string' ? lastUser.content : JSON.stringify(lastUser.content)).slice(0, 120)
        : '';
      console.log(`[responses ${reqId}] messages=${messages.length} roles=${messages.map(m => m.role).join('>')} preview=${JSON.stringify(preview)}`);
      let saveToPath = null;
      if (save_to) {
        const wsDir = workspace_dir || WORKSPACE_DIR;
        if (path.isAbsolute(save_to)) saveToPath = save_to;
        else if (wsDir) saveToPath = path.join(wsDir, save_to);
        else return res.status(400).json({ error: { message: 'save_to requires workspace_dir or WORKSPACE_DIR env', type: 'invalid_request_error' } });
      }
      const persistAssistant = buildPersist(req, sessionId => sessionId, messages, sessionsRepo);
      const authInfo = await refreshTokenIfNeeded();
      if (isTokenExpired(authInfo)) {
        console.error(`[responses ${reqId}] REJECT 401: Trae token expired`);
        return res.status(401).json({ error: { message: 'Trae token expired. Please restart Trae IDE to refresh.', type: 'auth_error' } });
      }
      const options = {};
      if (funcName) options.function = funcName;
      if (config_name) options.config_name = config_name;
      if (workspace_dir) options.workspace_dir = workspace_dir;
      options.workspace = extractWorkspace(req);
      for (const key of ['temperature', 'top_p', 'max_tokens', 'presence_penalty', 'frequency_penalty', 'stop', 'seed', 'n']) {
        if (body[key] !== undefined) options[key] = body[key];
      }
      const hasImageContent = messages.some(m => Array.isArray(m.content) && m.content.some(c => c.type === 'image' || c.type === 'image_url'));
      if (hasImageContent) {
        const currentConfig = resolveModelId(modelName);
        const modelEntry = MODEL_MAP[Object.keys(MODEL_MAP).find(k => MODEL_MAP[k].config_name === currentConfig)];
        if (!modelEntry?.multimodal) {
          const mmModel = findMultimodalModel(currentConfig);
          if (mmModel) {
            console.log(`[responses ${reqId}] Image content detected, switching to multimodal model: ${mmModel}`);
            options.config_name = mmModel;
          }
        }
      }
      const baseCtx = { reqId, respId, msgId, modelName, messages, options, persistAssistant, saveToPath, llmUtilsChat, chatCompletion, createAgentTask, parseLlmUtilsChatStream, parseAgentTaskStream, parseTraeStreamChunk, syncFileToOutput };
      if (isStream) await streamResponse(req, res, baseCtx);
      else await nonStreamResponse(res, baseCtx);
    } catch (err) {
      console.error(`[responses ${reqId}] FATAL:`, err);
      res.status(500).json({ error: { message: err.message, type: 'internal_error' } });
    }
  };
}

function buildPersist(req, _id, messages, sessionsRepo) {
  const sessionId = req.headers['x-session-id'];
  if (!sessionId || typeof sessionId !== 'string') return null;
  let persistAssistant = null;
  try {
    const lastUser = [...messages].reverse().find(m => m && m.role === 'user');
    if (lastUser && lastUser.content != null) {
      const userContent = typeof lastUser.content === 'string' ? lastUser.content : JSON.stringify(lastUser.content);
      const userMsg = sessionsRepo.addMessage(sessionId, { role: 'user', content: userContent });
      if (userMsg) {
        persistAssistant = (content, reasoning, tokenUsage) => {
          const fullContent = reasoning ? `[Thinking...]\n${reasoning}\n\n${content}` : content;
          sessionsRepo.addMessage(sessionId, {
            role: 'assistant',
            content: fullContent,
            tokensIn: (tokenUsage && tokenUsage.prompt_tokens) || 0,
            tokensOut: (tokenUsage && tokenUsage.completion_tokens) || 0,
          });
        };
      }
    }
  } catch (persistErr) {
    console.error('[persist] user message failed:', persistErr);
  }
  return persistAssistant;
}

async function streamResponse(req, res, ctx) {
  const {
    reqId, respId, msgId, modelName, messages, options, persistAssistant, saveToPath,
    llmUtilsChat, chatCompletion, parseLlmUtilsChatStream, parseAgentTaskStream, parseTraeStreamChunk, syncFileToOutput,
  } = ctx;
  const tag = `responses ${reqId || respId}`;
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('X-Accel-Buffering', 'no');
  let eventCount = 0;
  let failed = false;
  const sendEvent = (eventType, data) => {
    if (res.writableEnded) {
      console.warn(`[${tag}] skip write (ended): ${eventType}`);
      return;
    }
    eventCount += 1;
    if (eventType === 'response.failed' || eventType === 'response.completed' || eventCount <= 3) {
      console.log(`[${tag}] SSE #${eventCount} ${eventType}`);
    }
    res.write('event: ' + eventType + '\n');
    res.write('data: ' + JSON.stringify({ type: eventType, ...(data || {}) }) + '\n\n');
  };
  const failStream = (source, message, extra) => {
    if (failed || res.writableEnded) return;
    failed = true;
    const msg = message || 'unknown error';
    console.error(`[${tag}] response.failed source=${source} message=${msg}`, extra || '');
    sendEvent('response.failed', {
      response: {
        id: respId,
        object: 'response',
        status: 'failed',
        error: { message: msg, code: extra && extra.code, source },
      },
    });
    endStream();
  };
  sendEvent('response.created', {
    response: { id: respId, object: 'response', created_at: Math.floor(Date.now() / 1000), model: modelName, status: 'in_progress', output: [] },
  });
  let fullContent = '';
  let fullReasoning = '';
  let tokenUsage = null;
  let messageItemSent = false;
  let clientClosed = false;
  const startMessageItem = () => {
    if (messageItemSent) return;
    messageItemSent = true;
    sendEvent('response.output_item.added', { output_index: 0, item: { type: 'message', id: msgId, status: 'in_progress', role: 'assistant', content: [] } });
    sendEvent('response.content_part.added', { item_id: msgId, output_index: 0, content_index: 0, part: { type: 'output_text', text: '' } });
  };
  const flushItem = () => {
    if (!messageItemSent) return;
    sendEvent('response.output_text.done', { item_id: msgId, output_index: 0, content_index: 0, text: fullContent });
    sendEvent('response.output_item.done', { output_index: 0, item: { type: 'message', id: msgId, status: 'completed', role: 'assistant', content: [{ type: 'output_text', text: fullContent }] } });
    messageItemSent = false;
  };
  const finalize = (logId) => {
    flushItem();
    const usage = tokenUsage ? {
      input_tokens: tokenUsage.prompt_tokens || 0,
      output_tokens: tokenUsage.completion_tokens || 0,
      total_tokens: tokenUsage.total_tokens || 0,
    } : undefined;
    console.log(`[${tag}] complete content_len=${fullContent.length} reasoning_len=${fullReasoning.length} events=${eventCount}`);
    sendEvent('response.completed', { response: createResponsesResponse(respId, modelName, fullContent, fullReasoning, usage, 'completed') });
    if (logId) trafficLogger.finalizeLog(logId, { fullContent, fullReasoning, tokenUsage });
  };
  const endStream = () => {
    if (!res.writableEnded) { res.write('data: [DONE]\n\n'); res.end(); }
  };
  try {
    console.log(`[${tag}] calling llmUtilsChat model=${modelName}`);
    const result = await llmUtilsChat(messages, modelName, true, options);
    const logId = result.logId;
    console.log(`[${tag}] llmUtilsChat ok logId=${logId || '-'} hasBody=${!!result.body}`);
    if (result.body) {
      let buffer = '';
      let currentEventName = '';
      result.body.on('data', (chunk) => {
        try {
          buffer += chunk.toString();
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';
          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed) continue;
            const parsed = parseLlmUtilsChatStream(trimmed, currentEventName);
            if (!parsed) continue;
            if (parsed._type === 'event_name') {
              currentEventName = parsed.value;
              if (logId) trafficLogger.logResponseChunk(logId, currentEventName, null);
              continue;
            }
            if (logId) trafficLogger.logResponseChunk(logId, currentEventName, parsed);
            if (parsed.type === 'token_usage') {
              tokenUsage = parsed.data;
              if (logId) trafficLogger.logTokenUsage(logId, tokenUsage);
              continue;
            }
            if (parsed.type === 'done') {
              if (failed || res.writableEnded) return;
              if (persistAssistant) {
                try { persistAssistant(fullContent, fullReasoning, tokenUsage); }
                catch (e) { console.error('[persist] assistant (responses stream done) failed:', e); }
              }
              if (saveToPath && fullContent) {
                try {
                  const dir = path.dirname(saveToPath);
                  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
                  fs.writeFileSync(saveToPath, fullContent, 'utf-8');
                  syncFileToOutput(saveToPath);
                } catch (fileErr) {
                  console.error(`[file] Save failed: ${fileErr.message}`);
                }
              }
              finalize(logId);
              endStream();
              return;
            }
            if (parsed.type === 'text') {
              if (parsed.content) {
                startMessageItem();
                fullContent += parsed.content;
                if (logId) trafficLogger.logResponseContent(logId, parsed.content, null);
                sendEvent('response.output_text.delta', { item_id: msgId, output_index: 0, content_index: 0, delta: parsed.content });
              }
              if (parsed.reasoning) {
                // Accumulate only — do not emit non-standard response.reasoning.delta
                // (Codex/OpenAI Responses clients reject it and may surface response.failed).
                fullReasoning += parsed.reasoning;
                if (logId) trafficLogger.logResponseContent(logId, null, parsed.reasoning);
              }
            }
            if (parsed.type === 'error') {
              failStream('upstream_sse_error', parsed.message || 'unknown error', { code: parsed.code, raw: parsed });
            }
          }
        } catch (err) {
          console.error(`[${tag}] Error in data callback:`, err);
          if (logId) trafficLogger.logError(logId, err);
          try { result.body.destroy(); } catch (e) {}
          failStream('data_callback', err.message);
        }
      });
      result.body.on('end', () => {
        if (!res.writableEnded && !failed) {
          console.log(`[${tag}] upstream body end (no done event), content_len=${fullContent.length}`);
          if (persistAssistant) {
            try { persistAssistant(fullContent, fullReasoning, tokenUsage); }
            catch (e) { console.error('[persist] assistant (responses stream end) failed:', e); }
          }
          finalize(logId);
          endStream();
        }
      });
      result.body.on('error', (err) => {
        if (clientClosed) {
          console.warn(`[${tag}] upstream error after client close: ${err.message}`);
          return;
        }
        failStream('upstream_body_error', err.message);
      });
      req.on('close', () => {
        clientClosed = true;
        console.warn(`[${tag}] client closed early content_len=${fullContent.length} events=${eventCount} ended=${res.writableEnded}`);
        if (result.body && result.body.destroy) result.body.destroy();
      });
    } else {
      console.warn(`[${tag}] empty upstream body`);
      if (persistAssistant) {
        try { persistAssistant('', '', null); } catch (e) { console.error('[persist] assistant (responses empty body) failed:', e); }
      }
      finalize(logId);
      endStream();
    }
  } catch (llmErr) {
    console.log(`[${tag}] llmUtilsChat failed: ${llmErr.message}, falling back to chatCompletion`);
    try {
      const responseBody = await chatCompletion(messages, modelName, true, options);
      await new Promise((resolve, reject) => {
        let buffer = '';
        responseBody.on('data', (chunk) => {
          buffer += chunk.toString();
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';
          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed) continue;
            const parsed = parseAgentTaskStream(trimmed) || parseTraeStreamChunk(trimmed);
            if (!parsed) continue;
            if (parsed.done) return;
            if (parsed.content) {
              startMessageItem();
              fullContent += parsed.content;
              sendEvent('response.output_text.delta', { item_id: msgId, output_index: 0, content_index: 0, delta: parsed.content });
            }
          }
        });
        responseBody.on('end', resolve);
        responseBody.on('error', reject);
      });
      finalize(null);
      endStream();
    } catch (chatErr) {
      failStream('fallback_chatCompletion', chatErr.message, { llmErr: llmErr.message });
    }
  }
}

async function nonStreamResponse(res, ctx) {
  const {
    respId, modelName, messages, options, persistAssistant, saveToPath,
    llmUtilsChat, chatCompletion, createAgentTask, parseLlmUtilsChatStream, parseAgentTaskStream, parseTraeStreamChunk, syncFileToOutput,
  } = ctx;
  try {
    const result = await llmUtilsChat(messages, modelName, true, options);
    let fullContent = '';
    let fullReasoning = '';
    let tokenUsage = null;
    const upstreamLogId = result.logId;
    if (result.body) {
      await new Promise((resolve, reject) => {
        let buffer = '';
        let currentEventName = '';
        result.body.on('data', (chunk) => {
          buffer += chunk.toString();
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';
          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed) continue;
            const parsed = parseLlmUtilsChatStream(trimmed, currentEventName);
            if (!parsed) continue;
            if (parsed._type === 'event_name') {
              currentEventName = parsed.value;
              if (upstreamLogId) trafficLogger.logResponseChunk(upstreamLogId, currentEventName, null);
              continue;
            }
            if (upstreamLogId) trafficLogger.logResponseChunk(upstreamLogId, currentEventName, parsed);
            if (parsed.type === 'token_usage') {
              tokenUsage = parsed.data;
              if (upstreamLogId) trafficLogger.logTokenUsage(upstreamLogId, tokenUsage);
              continue;
            }
            if (parsed.type === 'done') continue;
            if (parsed.type === 'text' && parsed.content) {
              fullContent += parsed.content;
              if (upstreamLogId) trafficLogger.logResponseContent(upstreamLogId, parsed.content, null);
            }
            if (parsed.type === 'text' && parsed.reasoning) {
              fullReasoning += parsed.reasoning;
              if (upstreamLogId) trafficLogger.logResponseContent(upstreamLogId, null, parsed.reasoning);
            }
          }
        });
        result.body.on('end', resolve);
        result.body.on('error', reject);
      });
    }
    const usage = tokenUsage ? {
      input_tokens: tokenUsage.prompt_tokens || 0,
      output_tokens: tokenUsage.completion_tokens || 0,
      total_tokens: tokenUsage.total_tokens || 0,
    } : undefined;
    const response = createResponsesResponse(respId, modelName, fullContent, fullReasoning, usage, 'completed');
    if (upstreamLogId) trafficLogger.finalizeLog(upstreamLogId, { fullContent, fullReasoning, tokenUsage });
    if (saveToPath && fullContent) {
      try {
        const dir = path.dirname(saveToPath);
        if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
        fs.writeFileSync(saveToPath, fullContent, 'utf-8');
        syncFileToOutput(saveToPath);
      } catch (fileErr) {
        console.error(`[file] Save failed: ${fileErr.message}`);
      }
    }
    if (persistAssistant) {
      try { persistAssistant(fullContent, fullReasoning, tokenUsage); }
      catch (e) { console.error('[persist] assistant (responses non-stream) failed:', e); }
    }
    res.json(response);
  } catch (llmErr) {
    console.log(`[responses] non-stream failed: ${llmErr.message}, falling back`);
    let content = '';
    try {
      const responseBody = await chatCompletion(messages, modelName, true, options);
      await new Promise((resolve, reject) => {
        let buffer = '';
        responseBody.on('data', (chunk) => {
          buffer += chunk.toString();
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';
          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed) continue;
            const parsed = parseAgentTaskStream(trimmed) || parseTraeStreamChunk(trimmed);
            if (!parsed) continue;
            if (parsed.done) return;
            if (parsed.content) content += parsed.content;
          }
        });
        responseBody.on('end', resolve);
        responseBody.on('error', reject);
      });
    } catch (chatErr) {
      try {
        const agentResult = await createAgentTask(messages, modelName, true, options);
        if (agentResult.body) {
          await new Promise((resolve, reject) => {
            let buffer = '';
            agentResult.body.on('data', (chunk) => {
              buffer += chunk.toString();
              const lines = buffer.split('\n');
              buffer = lines.pop() || '';
              for (const line of lines) {
                const trimmed = line.trim();
                if (!trimmed) continue;
                const parsed = parseAgentTaskStream(trimmed) || parseTraeStreamChunk(trimmed);
                if (!parsed) continue;
                if (parsed.content) content += parsed.content;
              }
            });
            agentResult.body.on('end', resolve);
            agentResult.body.on('error', reject);
          });
        }
      } catch (agentErr) {
        throw new Error(`All endpoints failed: [llm] ${llmErr.message} [chat] ${chatErr.message} [agent] ${agentErr.message}`);
      }
    }
    if (persistAssistant) {
      try { persistAssistant(content, '', null); }
      catch (e) { console.error('[persist] assistant (responses fallback) failed:', e); }
    }
    const response = createResponsesResponse(respId, modelName, content, '', undefined, 'completed');
    res.json(response);
  }
}

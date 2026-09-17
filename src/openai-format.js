const { v4: uuidv4 } = require('./uuid');

function createOpenAIChatCompletion(id, model, content, finishReason, reasoning, usage) {
  const message = {
    role: 'assistant',
    content: content
  };
  if (reasoning) {
    message.reasoning_content = reasoning;
  }
  return {
    id: id || `chatcmpl-${uuidv4()}`,
    object: 'chat.completion',
    created: Math.floor(Date.now() / 1000),
    model: model,
    choices: [{
      index: 0,
      message: message,
      finish_reason: finishReason || 'stop'
    }],
    usage: usage || {
      prompt_tokens: 0,
      completion_tokens: 0,
      total_tokens: 0
    }
  };
}

function createOpenAIStreamChunk(id, model, delta, finishReason) {
  return {
    id: id || `chatcmpl-${uuidv4()}`,
    object: 'chat.completion.chunk',
    created: Math.floor(Date.now() / 1000),
    model: model,
    choices: [{
      index: 0,
      delta: delta,
      finish_reason: finishReason || null
    }]
  };
}

function createOpenAIModels(models) {
  return {
    object: 'list',
    data: models.map(m => ({
      id: m,
      object: 'model',
      created: Math.floor(Date.now() / 1000),
      owned_by: 'trae'
    }))
  };
}

function parseLlmUtilsChatSSE(rawBody) {
  const events = [];
  const lines = rawBody.split('\n');
  let currentEvent = null;

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) continue;

    if (trimmed.startsWith('event:')) {
      currentEvent = { event: trimmed.substring(6).trim(), data: '' };
      events.push(currentEvent);
    } else if (trimmed.startsWith('data:') && currentEvent) {
      currentEvent.data = trimmed.substring(5).trim();
    }
  }

  return events;
}

function parseLlmUtilsChatStream(rawLine, currentEventName) {
  if (!rawLine || rawLine.trim() === '') return null;

  if (rawLine.startsWith('event:')) {
    return { _type: 'event_name', value: rawLine.substring(6).trim() };
  }

  if (rawLine.startsWith('data:')) {
    const data = rawLine.substring(5).trim();
    if (data === '[DONE]') return { done: true };

    try {
      const parsed = JSON.parse(data);
      return normalizeLlmUtilsChunk(parsed, currentEventName);
    } catch (e) {
      return { raw: data };
    }
  }

  return null;
}

function normalizeLlmUtilsChunk(chunk, eventName) {
  if (!chunk || typeof chunk !== 'object') return chunk;

  if (eventName === 'output') {
    const result = { type: 'text' };

    // Old format: {"response": "..."}
    if (chunk.response) {
      const resp = chunk.response;
      if (resp.startsWith('Building prompt:') || resp.startsWith('Completed building prompt')) {
        return { type: 'progress', data: resp };
      }
      result.content = (result.content || '') + resp;
    }
    // New format (2026-05): {"type":"text","content":"..."}
    if (chunk.content) {
      result.content = (result.content || '') + chunk.content;
    }

    // Old format: {"reasoning_content": "..."}
    if (chunk.reasoning_content) {
      result.reasoning = (result.reasoning || '') + chunk.reasoning_content;
    }
    // New format (2026-05): {"reasoning": "..."}
    if (chunk.reasoning) {
      result.reasoning = (result.reasoning || '') + chunk.reasoning;
    }

    if (chunk.tool_calls) {
      result.tool_calls = chunk.tool_calls;
    }
    if (!result.content && !result.reasoning && !result.tool_calls) return null;
    return result;
  }

  if (eventName === 'done') {
    return { type: 'done', finish_reason: chunk.finish_reason || 'stop' };
  }

  if (eventName === 'error') {
    return {
      type: 'error',
      code: chunk.code,
      message: chunk.message,
      extra: chunk.extra
    };
  }

  if (eventName === 'token_usage') {
    return { type: 'token_usage', data: chunk };
  }

  if (eventName === 'extra_info') {
    return { type: 'extra_info', data: chunk };
  }

  if (eventName === 'metadata') {
    return { type: 'metadata', data: chunk };
  }

  if (eventName === 'timing_cost') {
    return { type: 'timing_cost', data: chunk };
  }

  if (eventName === 'progress_notice') {
    return { type: 'progress', data: chunk };
  }

  // Queue events - 排队相关事件
  if (eventName === 'queue_begin') {
    return { type: 'queue_begin', data: chunk };
  }

  if (eventName === 'request_wait_in_queue') {
    const position = chunk?.data?.position || chunk?.position || 0;
    if (position > 0) {
      // 只在位置是 10 的倍数或 < 20 时输出，减少刷屏
      if (position <= 20 || position % 50 === 0) {
        console.log(`[queue] Position: ${position} - waiting...`);
      }
    }
    return { type: 'queue_wait', position: position, data: chunk };
  }

  if (eventName === 'queue_end') {
    console.log(`[queue] Queue ended - request processing...`);
    return { type: 'queue_end', data: chunk };
  }

  return { type: eventName || 'unknown', data: chunk };
}

function llmUtilsChunkToOpenAI(chunk, id, model, includeReasoning) {
  if (!chunk) return null;

  if (chunk.type === 'done') {
    return createOpenAIStreamChunk(id, model, {}, chunk.finish_reason || 'stop');
  }

  if (chunk.type === 'text') {
    const delta = {};
    if (chunk.content) {
      delta.content = chunk.content;
    }
    if (includeReasoning && chunk.reasoning) {
      delta.reasoning_content = chunk.reasoning;
    }
    if (Object.keys(delta).length === 0) return null;
    return createOpenAIStreamChunk(id, model, delta, null);
  }

  if (chunk.type === 'error') {
    return createOpenAIStreamChunk(id, model, {
      content: `\n[Error ${chunk.code || ''}: ${chunk.message || 'unknown'}]`
    }, null);
  }

  return null;
}

function parseTraeStreamChunk(rawLine) {
  if (!rawLine || rawLine.trim() === '') return null;
  if (rawLine.startsWith('data: ')) {
    const data = rawLine.substring(6).trim();
    if (data === '[DONE]') return { done: true };
    try {
      return JSON.parse(data);
    } catch (e) {
      return { raw: data };
    }
  }
  if (rawLine.startsWith('event:')) return null;
  if (rawLine.startsWith('id:')) return null;
  if (rawLine.startsWith('retry:')) return null;
  try {
    return JSON.parse(rawLine);
  } catch (e) {
    return null;
  }
}

function traeChunkToOpenAI(chunk, id, model) {
  if (!chunk) return null;

  if (chunk.done) {
    return createOpenAIStreamChunk(id, model, {}, 'stop');
  }

  if (chunk.type === 'message' || chunk.type === 'text') {
    const content = chunk.data?.text || chunk.data?.content || chunk.data || '';
    if (content) {
      return createOpenAIStreamChunk(id, model, { content: String(content) }, null);
    }
    return null;
  }

  if (chunk.type === 'message_start' || chunk.type === 'start') {
    return createOpenAIStreamChunk(id, model, { role: 'assistant', content: '' }, null);
  }

  if (chunk.type === 'message_end' || chunk.type === 'end' || chunk.type === 'finish') {
    return createOpenAIStreamChunk(id, model, {}, 'stop');
  }

  if (chunk.type === 'error') {
    return createOpenAIStreamChunk(id, model, {
      content: `\n[Error: ${chunk.data?.message || chunk.message || 'unknown error'}]`
    }, null);
  }

  if (chunk.choices && chunk.choices.length > 0) {
    const choice = chunk.choices[0];
    const delta = {};
    if (choice.delta) {
      if (choice.delta.content) delta.content = choice.delta.content;
      if (choice.delta.role) delta.role = choice.delta.role;
    } else if (choice.message) {
      delta.content = choice.message.content || '';
      delta.role = choice.message.role || 'assistant';
    }
    return createOpenAIStreamChunk(id, model, delta, choice.finish_reason || null);
  }

  if (chunk.content !== undefined) {
    return createOpenAIStreamChunk(id, model, { content: chunk.content }, null);
  }

  if (chunk.message !== undefined) {
    return createOpenAIStreamChunk(id, model, { content: chunk.message || '' }, null);
  }

  if (chunk.text !== undefined) {
    return createOpenAIStreamChunk(id, model, { content: chunk.text || '' }, null);
  }

  if (chunk.delta !== undefined) {
    return createOpenAIStreamChunk(id, model, { content: chunk.delta }, null);
  }

  if (chunk.finish_reason) {
    return createOpenAIStreamChunk(id, model, {}, 'stop');
  }

  return null;
}

function parseAgentTaskStream(rawLine) {
  if (!rawLine || rawLine.trim() === '') return null;
  if (rawLine.startsWith('data: ')) {
    const data = rawLine.substring(6).trim();
    if (data === '[DONE]') return { done: true };
    try {
      const parsed = JSON.parse(data);
      return normalizeAgentTaskChunk(parsed);
    } catch (e) {
      return { raw: data };
    }
  }
  if (rawLine.startsWith('event:')) return null;
  if (rawLine.startsWith('id:')) return null;
  try {
    const parsed = JSON.parse(rawLine);
    return normalizeAgentTaskChunk(parsed);
  } catch (e) {
    return null;
  }
}

function normalizeAgentTaskChunk(chunk) {
  if (!chunk || typeof chunk !== 'object') return chunk;

  if (chunk.event_type) {
    switch (chunk.event_type) {
      case 'message_start':
      case 'session_start':
      case 'task_start':
        return { type: 'message_start', data: chunk };
      case 'text_delta':
      case 'content_delta':
      case 'message_delta':
        return {
          type: 'text',
          data: chunk.data?.text || chunk.data?.content || chunk.text || chunk.content || ''
        };
      case 'tool_call':
      case 'function_call':
        return { type: 'tool_call', data: chunk.data || chunk };
      case 'tool_result':
      case 'function_result':
        return { type: 'tool_result', data: chunk.data || chunk };
      case 'message_end':
      case 'session_end':
      case 'task_end':
      case 'finish':
        return { type: 'message_end', data: chunk };
      case 'error':
        return { type: 'error', data: chunk.data || chunk };
      default:
        return chunk;
    }
  }

  if (chunk.type) return chunk;

  if (chunk.text || chunk.content || chunk.delta) {
    return {
      type: 'text',
      data: chunk.text || chunk.content || chunk.delta || ''
    };
  }

  return chunk;
}

// ==================== OpenAI Responses API (v1/responses) ====================

/**
 * Convert Responses API `input` field to OpenAI Chat messages.
 * Accepts either an array of messages/items or a plain string.
 * Supports both Chat-style message objects and Responses `response.output_item` shapes.
 */
function responsesInputToMessages(input, instructions) {
  let messages = [];

  if (instructions) {
    messages.push({ role: 'system', content: instructions });
  }

  if (input == null) {
    return messages;
  }

  if (typeof input === 'string') {
    messages.push({ role: 'user', content: input });
    return messages;
  }

  if (!Array.isArray(input)) {
    messages.push({ role: 'user', content: JSON.stringify(input) });
    return messages;
  }

  for (const item of input) {
    if (!item || typeof item !== 'object') continue;

    // Already a chat-style message: { role, content } (Codex sends this with input_text parts)
    if (item.role && item.content !== undefined) {
      messages.push({
        role: normalizeResponsesRole(item.role),
        content: normalizeResponsesContent(item.content),
      });
      continue;
    }

    // Responses output_item shape: { type: 'message', role, content: [{type:'output_text'|'input_text', text}] }
    if (item.type === 'message' && item.role) {
      messages.push({
        role: normalizeResponsesRole(item.role),
        content: normalizeResponsesContent(item.content),
      });
      continue;
    }

    // Responses input_text item: { type: 'input_text'|'input_image', ... }
    if (item.type === 'input_text') {
      messages.push({ role: normalizeResponsesRole(item.role || 'user'), content: item.text || '' });
      continue;
    }
    if (item.type === 'input_image') {
      const content = item.image_url || item.url || '';
      messages.push({ role: normalizeResponsesRole(item.role || 'user'), content: [{ type: 'image_url', image_url: { url: content } }] });
      continue;
    }

    // Codex / Responses function calling items
    if (item.type === 'function_call') {
      const args = typeof item.arguments === 'string' ? item.arguments : JSON.stringify(item.arguments || {});
      messages.push({
        role: 'assistant',
        content: `[function_call name=${item.name || ''} call_id=${item.call_id || ''}]\n${args}`,
      });
      continue;
    }
    if (item.type === 'custom_tool_call') {
      const input = typeof item.input === 'string' ? item.input : JSON.stringify(item.input || {});
      messages.push({
        role: 'assistant',
        content: `[custom_tool_call name=${item.name || ''} call_id=${item.call_id || ''}]\n${input}`,
      });
      continue;
    }
    if (item.type === 'function_call_output' || item.type === 'custom_tool_call_output') {
      const out = typeof item.output === 'string' ? item.output : JSON.stringify(item.output ?? '');
      messages.push({
        role: 'user',
        content: `[${item.type} call_id=${item.call_id || ''}]\n${out}`,
      });
      continue;
    }

    // Fallback: try role/content
    if (item.role) {
      messages.push({
        role: normalizeResponsesRole(item.role),
        content: normalizeResponsesContent(item.content || item.text || ''),
      });
    }
  }

  return messages;
}

/** Map OpenAI Responses roles Trae does not accept. */
function normalizeResponsesRole(role) {
  if (role === 'developer') return 'system';
  return role || 'user';
}

/** Flatten Responses content parts (input_text/output_text) to string or Trae-safe parts. */
function normalizeResponsesContent(content) {
  if (content == null) return '';
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return String(content);
  const texts = [];
  const parts = [];
  let hasNonText = false;
  for (const c of content) {
    if (typeof c === 'string') {
      texts.push(c);
      parts.push({ type: 'text', text: c });
      continue;
    }
    if (!c || typeof c !== 'object') continue;
    if (c.type === 'input_text' || c.type === 'output_text' || c.type === 'text') {
      const t = c.text || '';
      texts.push(t);
      parts.push({ type: 'text', text: t });
      continue;
    }
    if (c.type === 'image_url' || c.type === 'input_image' || c.type === 'image') {
      hasNonText = true;
      if (c.type === 'input_image') {
        parts.push({ type: 'image_url', image_url: { url: c.image_url || c.url || '' } });
      } else {
        parts.push(c);
      }
      continue;
    }
    if (c.text) {
      texts.push(c.text);
      parts.push({ type: 'text', text: c.text });
    }
  }
  return hasNonText ? parts : texts.join('\n');
}

/**
 * Build a non-streaming Responses API response object.
 */
function createResponsesResponse(id, model, content, reasoning, usage, status, extraOutput) {
  const text = content || '';
  const respId = id || `resp_${uuidv4().replace(/-/g, '').substring(0, 24)}`;
  const output = [];
  if (text) {
    output.push({
      type: 'message',
      id: `msg_${uuidv4().replace(/-/g, '').substring(0, 24)}`,
      status: status || 'completed',
      role: 'assistant',
      content: [{ type: 'output_text', text: text }],
    });
  }
  if (Array.isArray(extraOutput) && extraOutput.length) {
    output.push(...extraOutput);
  }
  if (output.length === 0) {
    output.push({
      type: 'message',
      id: `msg_${uuidv4().replace(/-/g, '').substring(0, 24)}`,
      status: status || 'completed',
      role: 'assistant',
      content: [{ type: 'output_text', text: '' }],
    });
  }
  return {
    id: respId,
    object: 'response',
    created_at: Math.floor(Date.now() / 1000),
    model: model,
    status: status || 'completed',
    output,
    usage: usage || {
      input_tokens: 0,
      output_tokens: 0,
      total_tokens: 0
    }
  };
}

const TOOL_CALL_XML_RE = /<tool_?call\b[^>]*>\s*([\s\S]*?)\s*<\/tool_?call\s*>/gi;
const TOOL_CALL_OPEN_RE = /<tool_?call\b[^>]*>/gi;
const FUNCTION_CALL_HEADER_RE = /\[function_call\s+name=([^\s\]]+)[^\]]*\]/gi;

function extractBalancedJson(text, from = 0) {
  const start = String(text).indexOf('{', from);
  if (start < 0) return null;
  let depth = 0;
  let inStr = false;
  let esc = false;
  for (let i = start; i < text.length; i++) {
    const c = text[i];
    if (inStr) {
      if (esc) esc = false;
      else if (c === '\\') esc = true;
      else if (c === '"') inStr = false;
      continue;
    }
    if (c === '"') { inStr = true; continue; }
    if (c === '{') depth += 1;
    else if (c === '}') {
      depth -= 1;
      if (depth === 0) return { json: text.slice(start, i + 1), start, end: i + 1 };
    }
  }
  return null;
}

function normalizeOpenAIToolDef(tool) {
  if (!tool || typeof tool !== 'object') return null;
  if (tool.type === 'function' && tool.function) {
    return {
      name: tool.function.name,
      description: tool.function.description || '',
      parameters: tool.function.parameters || tool.function.input_schema || {},
      kind: 'function',
    };
  }
  if (tool.type === 'custom' || tool.type === 'freeform') {
    return {
      name: tool.name,
      description: tool.description || '',
      parameters: tool.parameters || tool.input_schema || {},
      kind: 'custom',
    };
  }
  if (tool.name) {
    return {
      name: tool.name,
      description: tool.description || '',
      parameters: tool.parameters || tool.input_schema || {},
      kind: tool.type === 'custom' ? 'custom' : 'function',
    };
  }
  return null;
}

function buildToolKindMap(tools) {
  const map = Object.create(null);
  for (const t of tools || []) {
    const d = normalizeOpenAIToolDef(t);
    if (d && d.name) map[d.name] = d.kind || 'function';
  }
  return map;
}

function buildResponsesToolPrompt(tools) {
  const defs = (tools || []).map(normalizeOpenAIToolDef).filter(Boolean);
  if (!defs.length) return '';
  const lines = defs.map((d) => {
    const props = d.parameters && d.parameters.properties
      ? Object.keys(d.parameters.properties).slice(0, 12).join(', ')
      : '';
    const desc = (d.description || '').replace(/\s+/g, ' ').slice(0, 160);
    return `- ${d.name}${props ? `(${props})` : ''}: ${desc}`;
  }).join('\n');
  const exampleName = defs[0]?.name || 'shell';
  return `You have tools. When you need a tool, output ONLY this XML (must close the tag):

<tool_call>
{"name":"${exampleName}","arguments":{"key":"value"}}
</tool_call>

Critical:
- Always close with </tool_call>
- "arguments" MUST be a JSON object (never an array, never a string)
- Use ONLY tool names from the list below (exact spelling)
- Do NOT narrate tool calls in prose; do NOT invent results

Available tools:
${lines}`;
}

function coerceToolArguments(args) {
  if (typeof args === 'string') {
    try { args = JSON.parse(args); } catch (_) { args = { value: args }; }
  }
  if (Array.isArray(args)) args = { command: args };
  if (!args || typeof args !== 'object' || Array.isArray(args)) args = {};
  if (typeof args.cmd === 'string' && args.command == null) {
    args = { ...args, command: args.cmd };
  }
  return args;
}

function resolveToolName(name, known) {
  if (!name) return null;
  if (!known || known.has(name)) return name;
  const aliases = {
    exec_command: ['shell', 'container.exec', 'local_shell'],
    shell_command: ['shell'],
    run_terminal_cmd: ['shell'],
    run_command: ['shell'],
  };
  for (const alt of (aliases[name] || [])) {
    if (known.has(alt)) return alt;
  }
  return name;
}

function pushCall(calls, name, args, known) {
  const resolved = resolveToolName(name, known);
  if (!resolved) return;
  calls.push({ name: resolved, arguments: coerceToolArguments(args) });
}

function stripToolCallXml(text) {
  if (!text) return '';
  let out = String(text);
  out = out.replace(TOOL_CALL_XML_RE, '');
  // Unclosed <tool_call>{...}
  let guard = 0;
  while (guard++ < 20) {
    const m = out.match(/<tool_?call\b[^>]*>/i);
    if (!m) break;
    const idx = m.index;
    const bal = extractBalancedJson(out, idx + m[0].length);
    if (!bal) {
      out = out.slice(0, idx).trimEnd();
      break;
    }
    let after = bal.end;
    const close = out.slice(after).match(/^\s*<\/tool_?call\s*>/i);
    if (close) after += close[0].length;
    out = out.slice(0, idx) + out.slice(after);
  }
  // [function_call name=...]{...}
  guard = 0;
  while (guard++ < 20) {
    const m = out.match(/\[function_call\s+name=[^\]]+\]/i);
    if (!m) break;
    const idx = m.index;
    const bal = extractBalancedJson(out, idx + m[0].length);
    if (!bal) {
      out = out.slice(0, idx).trimEnd();
      break;
    }
    out = out.slice(0, idx) + out.slice(bal.end);
  }
  out = out.replace(/```(?:json)?\s*\{[\s\S]*?"name"\s*:\s*"[^"]+"[\s\S]*?\}\s*```/gi, '');
  out = out.replace(/<\/?think>/gi, '');
  return out.trim();
}

function extractResponsesToolCalls(text, nativeToolCalls, knownToolNames) {
  const calls = [];
  const known = Array.isArray(knownToolNames) && knownToolNames.length
    ? new Set(knownToolNames)
    : null;
  const seen = new Set();

  const addParsed = (obj) => {
    if (!obj || typeof obj !== 'object') return;
    const name = obj.name || obj.tool || obj.function?.name;
    let args = obj.arguments ?? obj.parameters ?? obj.input ?? obj.function?.arguments;
    if (args == null && (obj.cmd != null || obj.command != null || obj.workdir != null)) {
      args = obj;
    }
    if (args == null) args = {};
    const key = `${name}::${JSON.stringify(coerceToolArguments(args))}`;
    if (!name || seen.has(key)) return;
    seen.add(key);
    pushCall(calls, name, args, known);
  };

  if (text) {
    TOOL_CALL_XML_RE.lastIndex = 0;
    let m;
    while ((m = TOOL_CALL_XML_RE.exec(text)) !== null) {
      try { addParsed(JSON.parse((m[1] || '').trim())); }
      catch (e) { console.warn('[responses] bad tool_call XML:', e.message); }
    }

    TOOL_CALL_OPEN_RE.lastIndex = 0;
    let om;
    while ((om = TOOL_CALL_OPEN_RE.exec(text)) !== null) {
      const bal = extractBalancedJson(text, om.index + om[0].length);
      if (!bal) continue;
      try { addParsed(JSON.parse(bal.json)); } catch (_) {}
    }

    FUNCTION_CALL_HEADER_RE.lastIndex = 0;
    let hm;
    while ((hm = FUNCTION_CALL_HEADER_RE.exec(text)) !== null) {
      const bal = extractBalancedJson(text, hm.index + hm[0].length);
      if (!bal) {
        pushCall(calls, hm[1], {}, known);
        continue;
      }
      try {
        const parsed = JSON.parse(bal.json);
        if (parsed.name) addParsed(parsed);
        else addParsed({ name: hm[1], arguments: parsed });
      } catch (_) {
        pushCall(calls, hm[1], {}, known);
      }
    }

    const fenceRe = /```(?:json)?\s*(\{[\s\S]*?\})\s*```/gi;
    let fm;
    while ((fm = fenceRe.exec(text)) !== null) {
      try {
        const parsed = JSON.parse(fm[1]);
        if (parsed.name || parsed.tool) addParsed(parsed);
      } catch (_) {}
    }
  }

  if (Array.isArray(nativeToolCalls)) {
    for (const tc of nativeToolCalls) {
      addParsed({
        name: tc.name || tc.function?.name,
        arguments: tc.arguments ?? tc.function?.arguments ?? tc.input ?? {},
      });
    }
  }
  return calls;
}

function createResponsesStreamEvent(type, data) {
  return { type, ...(data || {}) };
}

/**
 * Build a response.output_text.delta streaming event.
 */
function createResponsesTextDeltaEvent(itemId, contentIndex, delta) {
  return {
    type: 'response.output_text.delta',
    item_id: itemId,
    output_index: 0,
    content_index: contentIndex || 0,
    delta: delta || ''
  };
}

module.exports = {
  createOpenAIChatCompletion,
  createOpenAIStreamChunk,
  createOpenAIModels,
  createResponsesResponse,
  createResponsesStreamEvent,
  createResponsesTextDeltaEvent,
  responsesInputToMessages,
  buildResponsesToolPrompt,
  buildToolKindMap,
  extractResponsesToolCalls,
  stripToolCallXml,
  extractBalancedJson,
  parseLlmUtilsChatSSE,
  parseLlmUtilsChatStream,
  normalizeLlmUtilsChunk,
  llmUtilsChunkToOpenAI,
  parseTraeStreamChunk,
  parseAgentTaskStream,
  normalizeAgentTaskChunk,
  traeChunkToOpenAI
};

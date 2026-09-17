const { EventEmitter } = require('events');
const { Readable } = require('stream');
const createResponsesHandler = require('../src/responses-handler');

// Mock SSE mimicking the Trae llm_utils_chat stream.
const SSE = [
  'event: output',
  'data: {"content":"Hi"}',
  '',
  'event: token_usage',
  'data: {"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}',
  '',
  'event: done',
  'data: {}',
  '',
].join('\n');

function makeFakeRes() {
  let captured = '';
  return {
    writableEnded: false,
    setHeader() {},
    write(c) { captured += c; },
    end() { this.writableEnded = true; },
    json(o) { captured += JSON.stringify(o); this.writableEnded = true; },
    get captured() { return captured; },
  };
}

function makeParseLlmUtilsChatStream() {
  return (line, currentEventName) => {
    if (line.startsWith('event:')) return { _type: 'event_name', value: line.slice(6).trim() };
    if (line.startsWith('data:')) {
      try {
        const j = JSON.parse(line.slice(5).trim());
        // Mimic normalizeLlmUtilsChunk for the 'output' event: wrap raw content as text.
        if (currentEventName === 'output' && j.content) return { type: 'text', content: j.content };
        if (currentEventName === 'token_usage') return { type: 'token_usage', data: j };
        if (currentEventName === 'done') return { type: 'done', finish_reason: j.finish_reason || 'stop' };
        return j;
      } catch (e) { return null; }
    }
    return null;
  };
}

function makeDeps(overrides = {}) {
  return {
    responsesInputToMessages: (i) => [{ role: 'user', content: i }],
    resolveModelId: () => 'm',
    MODEL_MAP: {},
    findMultimodalModel: () => null,
    refreshTokenIfNeeded: async () => ({}),
    isTokenExpired: () => false,
    llmUtilsChat: async () => ({ body: Readable.from([SSE]), logId: null }),
    chatCompletion: async () => ({ body: null }),
    createAgentTask: async () => ({ body: null }),
    parseLlmUtilsChatStream: makeParseLlmUtilsChatStream(),
    parseAgentTaskStream: () => null,
    parseTraeStreamChunk: () => null,
    sessionsRepo: { addMessage: () => null },
    extractWorkspace: () => 'default',
    WORKSPACE_DIR: '',
    syncFileToOutput: () => {},
    ...overrides,
  };
}

async function testStreaming() {
  const handler = createResponsesHandler(makeDeps());
  const res = makeFakeRes();
  const req = Object.assign(new EventEmitter(), { body: { input: 'hello', stream: true }, headers: {} });
  await handler(req, res);
  await new Promise(r => setTimeout(r, 100));
  const out = res.captured;
  if (!out.includes('response.created')) throw new Error('missing response.created');
  if (!out.includes('response.output_text.delta')) throw new Error('missing delta event');
  if (!out.includes('"delta":"Hi"')) throw new Error('missing delta content');
  if (!out.includes('response.output_item.done')) throw new Error('missing output_item.done');
  if (!out.includes('response.completed')) throw new Error('missing response.completed');
  if (!out.includes('data: [DONE]')) throw new Error('missing [DONE]');
  console.log('[test] streaming OK');
}

async function testNonStreaming() {
  const handler = createResponsesHandler(makeDeps());
  const res = makeFakeRes();
  const req = Object.assign(new EventEmitter(), { body: { input: 'hello', stream: false }, headers: {} });
  await handler(req, res);
  const parsed = JSON.parse(res.captured);
  if (parsed.object !== 'response') throw new Error('wrong object: ' + parsed.object);
  if (parsed.status !== 'completed') throw new Error('wrong status');
  if (!parsed.output || !parsed.output[0]) throw new Error('missing output');
  const text = parsed.output[0].content.find(c => c.type === 'output_text');
  if (!text || text.text !== 'Hi') throw new Error('wrong text content: ' + JSON.stringify(text));
  if (!parsed.usage || parsed.usage.input_tokens !== 1) throw new Error('wrong usage');
  console.log('[test] non-streaming OK');
}

async function main() {
  await testStreaming();
  await testNonStreaming();
  console.log('All responses-handler tests passed.');
}

main().catch(e => { console.error('TEST FAILED:', e); process.exit(1); });

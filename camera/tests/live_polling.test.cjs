const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../static/live_polling.js'), 'utf8');
function client(fetch) {
  const context = {window:{}, fetch, AbortController, setTimeout, clearTimeout};
  vm.runInNewContext(source,context);
  return context.window.LivePolling;
}
test('hung headers time out and subsequent requests recover', async () => {
  let requests=0, signal;
  const api=client((url, options) => {
    signal=options.signal;
    return ++requests===1 ? new Promise(()=>{}) : Promise.resolve({ok:true,json:async()=>({online:true})});
  });
  await assert.rejects(api.fetchJSON('/state',{timeoutMs:15}),/超时/);
  assert.equal(signal.aborted,true);
  assert.equal((await api.fetchJSON('/state')).online,true);
});
test('a body that never completes also times out', async () => {
  const api=client(async()=>({ok:true,json:()=>new Promise(()=>{})}));
  await assert.rejects(api.fetchJSON('/state',{timeoutMs:15}),/超时/);
});
test('failed optional request does not block another independent source', async () => {
  const api=client(async url => url==='/hardware'
    ? {ok:true,json:()=>new Promise(()=>{})} : {ok:true,json:async()=>({sessions:[]})});
  const results=await Promise.allSettled([
    api.fetchJSON('/hardware',{timeoutMs:15}),api.fetchJSON('/events'),
  ]);
  assert.equal(results[0].status,'rejected');
  assert.equal(results[1].status,'fulfilled');
});

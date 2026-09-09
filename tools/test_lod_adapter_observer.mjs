// CPU-only fake WebGPU objects; no browser, adapter, device or renderer is opened.
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import test from 'node:test';

const source = await readFile(new URL('../www/lod-adapter-observer.js', import.meta.url));
const {observeRendererAdapter} = await import(`data:text/javascript;base64,${source.toString('base64')}`);

test('observer preserves actual objects/options and makes no independent GPU requests', async () => {
  const records = [];
  let adapters = 0, devices = 0;
  const adapterOptions = {powerPreference: 'high-performance'};
  const deviceOptions = {requiredFeatures: ['indirect-first-instance'], requiredLimits: {maxBindGroups: 4}};
  const device = {limits: {maxBufferSize: 2**30, maxStorageBufferBindingSize: 2**28, maxStorageBuffersPerShaderStage: 8}};
  const adapter = {info: {vendor: 'synthetic', architecture: 'test', device: '', description: '', isFallbackAdapter: false},
    requestDevice(options) { assert.equal(this, adapter); assert.equal(options, deviceOptions); devices++; return Promise.resolve(device); }};
  const gpu = {requestAdapter(options) { assert.equal(this, gpu); assert.equal(options, adapterOptions); adapters++; return Promise.resolve(adapter); }};
  observeRendererAdapter(gpu, 'a'.repeat(64), record => records.push(record));
  assert.equal(adapters, 0);
  assert.equal(devices, 0);
  assert.equal(await gpu.requestAdapter(adapterOptions), adapter);
  assert.equal(await adapter.requestDevice(deviceOptions), device);
  assert.equal(adapters, 1);
  assert.equal(devices, 1);
  assert.deepEqual(records.map(record => record.kind), ['browser_adapter', 'browser_device']);
  assert.equal(records[0].request_id, records[1].request_id);
  assert.equal(records[0].info.is_fallback_adapter, false);
  assert.equal(records[1].limits.maxBufferSize, 2**30);
});

test('null adapter and request errors remain failures from the original request', async () => {
  const records = [];
  const gpu = {requestAdapter() { return Promise.resolve(null); }};
  observeRendererAdapter(gpu, 'b'.repeat(64), record => records.push(record));
  assert.equal(await gpu.requestAdapter(), null);
  assert.equal(records[0].status, 'null');
  const error = new Error('synthetic rejection');
  const rejected = {requestAdapter() { return Promise.reject(error); }};
  observeRendererAdapter(rejected, 'b'.repeat(64), record => records.push(record));
  await assert.rejects(rejected.requestAdapter(), value => value === error);
  assert.equal(records[1].status, 'rejected');
});

test('device rejection is linked to its actual adapter without a successful-device claim', async () => {
  const records = [];
  const adapter = {info: {}, requestDevice() { throw new Error('synthetic device failure'); }};
  const gpu = {requestAdapter() { return Promise.resolve(adapter); }};
  observeRendererAdapter(gpu, 'c'.repeat(64), record => records.push(record));
  await gpu.requestAdapter();
  assert.throws(() => adapter.requestDevice(), /synthetic device failure/);
  assert.equal(records[0].info.is_fallback_adapter, null);
  assert.equal(records[1].status, 'rejected');
  assert.equal(records[1].request_id, records[0].request_id);
  assert.equal(records[1].limits, null);
});

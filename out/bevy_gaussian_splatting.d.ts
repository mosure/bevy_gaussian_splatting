/* tslint:disable */
/* eslint-disable */

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly main: (a: number, b: number) => number;
    readonly wasm_bindgen_79ec2e84eadfdadb___convert__closures_____invoke___wasm_bindgen_79ec2e84eadfdadb___JsValue__core_9cd9b4d2a02a8c45___result__Result_____wasm_bindgen_79ec2e84eadfdadb___JsError___true_: (a: number, b: number, c: any) => [number, number];
    readonly wasm_bindgen_79ec2e84eadfdadb___convert__closures_____invoke___js_sys_a45668d48711c2c8___Array__web_sys_2475a62cca0246b7___features__gen_ResizeObserver__ResizeObserver______true_: (a: number, b: number, c: any, d: any) => void;
    readonly wasm_bindgen_79ec2e84eadfdadb___convert__closures_____invoke___wasm_bindgen_79ec2e84eadfdadb___JsValue______true_: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen_79ec2e84eadfdadb___convert__closures_____invoke___wasm_bindgen_79ec2e84eadfdadb___JsValue______true__1_: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen_79ec2e84eadfdadb___convert__closures_____invoke___wasm_bindgen_79ec2e84eadfdadb___JsValue______true__2_: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen_79ec2e84eadfdadb___convert__closures_____invoke___wasm_bindgen_79ec2e84eadfdadb___JsValue______true__2__5: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen_79ec2e84eadfdadb___convert__closures_____invoke___wasm_bindgen_79ec2e84eadfdadb___JsValue______true__2__6: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen_79ec2e84eadfdadb___convert__closures_____invoke___wasm_bindgen_79ec2e84eadfdadb___JsValue______true__2__7: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen_79ec2e84eadfdadb___convert__closures_____invoke___wasm_bindgen_79ec2e84eadfdadb___JsValue______true__1__8: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen_79ec2e84eadfdadb___convert__closures_____invoke___wasm_bindgen_79ec2e84eadfdadb___JsValue______true__2__9: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen_79ec2e84eadfdadb___convert__closures_____invoke___wasm_bindgen_79ec2e84eadfdadb___JsValue______true__2__10: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen_79ec2e84eadfdadb___convert__closures_____invoke___wasm_bindgen_79ec2e84eadfdadb___JsValue______true__2__11: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen_79ec2e84eadfdadb___convert__closures_____invoke___wasm_bindgen_79ec2e84eadfdadb___JsValue______true__2__12: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen_79ec2e84eadfdadb___convert__closures_____invoke_______true_: (a: number, b: number) => void;
    readonly __wbindgen_malloc_command_export: (a: number, b: number) => number;
    readonly __wbindgen_realloc_command_export: (a: number, b: number, c: number, d: number) => number;
    readonly __externref_table_alloc_command_export: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_exn_store_command_export: (a: number) => void;
    readonly __wbindgen_free_command_export: (a: number, b: number, c: number) => void;
    readonly __wbindgen_destroy_closure_command_export: (a: number, b: number) => void;
    readonly __externref_table_dealloc_command_export: (a: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;

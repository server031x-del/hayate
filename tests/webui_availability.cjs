// Exercise the actual UI controller without starting a browser/server or GPU.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const html = fs.readFileSync('hayate/webui/static/index.html', 'utf8');
const elements = new Map();
class Element {
  constructor() { this.value = ''; this.disabled = false; this.hidden = false; this.children = []; this.classList = {toggle() {}}; }
  setAttribute() {}
  removeAttribute() {}
  toggleAttribute(name, value) { this[name] = value; }
  closest() { return this.card || this; }
  querySelector() { return this.children[0] || null; }
  append(item) { this.children.push(item); }
  contains(item) { return this === item.card; }
}
for (const match of html.matchAll(/id="([^"]+)"/g)) elements.set('#' + match[1], new Element());
const radios = [...html.matchAll(/name="profile" value="([^"]+)"([^>]*)/g)].map(match => {
  const radio = new Element(); radio.value = match[1]; radio.disabled = match[2].includes('disabled'); radio.card = new Element(); return radio;
});
let checked = radios.find(r => r.value === 'fast_sage_detail');
for (const radio of radios) Object.defineProperty(radio, 'checked', {get: () => checked === radio, set: v => { if (v) checked = radio; }});
const options = ['auto', 't2va', 'fl2va', 'ref2va'].map(value => Object.assign(new Element(), {value}));
const document = {
  querySelector(selector) {
    if (selector === 'input[name="profile"]:checked') return checked;
    const m = selector.match(/input\[name="profile"\]\[value="([^"]+)"\]/);
    if (m) return radios.find(r => r.value === m[1]);
    if (selector === '#task option:checked') return options.find(o => o.value === elements.get('#task').value);
    return elements.get(selector) || null;
  },
  querySelectorAll(selector) {
    if (selector === 'input[name="profile"]') return radios;
    if (selector === '.profile-card') return radios.map(r => r.card);
    if (selector === '#task option') return options;
    return [];
  },
  createElement() { return new Element(); },
};
const context = vm.createContext({document, console});
vm.runInContext(fs.readFileSync('hayate/webui/static/app.js', 'utf8').replace(/initialize\(\);\s*$/, ''), context);
const run = script => vm.runInContext(script, context);
run('updateModelAvailability()');
assert(radios.every(r => r.disabled), 'Unknown inventory must not enable profiles');
run('state.modelSetup = {assets: STANDARD_MODEL_IDS.map(id => ({id,status:"verified"}))}; updateModelAvailability()');
assert(!radios.find(r => r.value === 'fast_sage_detail').disabled);
assert(radios.find(r => r.value === 'pdd').disabled);
assert(elements.get('#pdd').disabled);
run('state.modelSetup.assets.push(...PDD_MODEL_IDS.map(id => ({id,status:"verified"}))); updateModelAvailability()');
assert(!elements.get('#pdd').disabled);
run('state.modelSetup.comfy_fasth3 = {ready:true,message:"ready"}; updateModelAvailability()');
checked = radios.find(r => r.value === 'comfy_fasth3');
run('applyProfile("comfy_fasth3"); updateModelAvailability()');
assert(elements.get('#steps').disabled);
assert(!elements.get('#vsaKeep').disabled);
assert(!elements.get('#comfyOptions').hidden);
assert.equal(elements.get('#task').value, 't2va');
checked = radios.find(r => r.value === 'fast');
run('applyProfile("fast"); updateModelAvailability()');
assert(!elements.get('#steps').disabled, 'Switching back restores standard controls');
assert(elements.get('#comfyOptions').hidden);
run('state.modelSetup = null; updateModelAvailability()');
assert(radios.every(r => r.disabled), 'A failed refresh must not keep stale availability');
console.log('UI availability transitions passed');
run('state.imageAsset = {id:"first"}; state.lastImageAsset = {id:"last"}');
assert.equal(run('generationPayload().image_asset_id'), 'first');
assert.equal(run('generationPayload().last_image_asset_id'), 'last');

assert(/id="prompt"[^>]*><\/textarea>/.test(html), 'Prompt starts empty');
assert(html.includes('id="samplePrompt"'));
assert(html.includes('id="lastImageFile"'));
console.log('Composer payload and empty initial prompt passed');

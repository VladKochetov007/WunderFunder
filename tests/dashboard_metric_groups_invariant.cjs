const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const file = process.argv[2] || require('node:path').resolve(__dirname, '../src/wnn/dashboard/static/dashboard.js');
const source = fs.readFileSync(file, 'utf8');
const functions = [...source.matchAll(/^function\s+\w+\([^\n]*\)\s*\{[\s\S]*?^\}/gm)].map(match => match[0]);
assert.ok(functions.some(body => body.startsWith('function metricGroups(')), 'metricGroups must be inspectable');
const keys = [
 'valid/z/wp', 'valid/z/wmse', 'valid/z/raw_mean/t0', 'valid/z/raw_std/t0',
 'valid/z/prediction_clipped_fraction/t0', 'valid/z/prediction_clipped_count/t0',
 'valid/z/prediction_clipped_weight_mass/t0',
 'valid/target/raw_mean/t0', 'valid/target/official_weight_mass/t0',
 'train/objective_wmse_raw', 'train/objective_correlation_loss_raw',
 'train/composite_wmse', 'train/blend_alpha',
 'train/objective_weight/wmse', 'train/objective_weight/correlation', 'train/loss',
 'train/objective_correlation_surrogate/t0', 'train/objective_correlation_surrogate/t1',
 'diagnostics/missing_output_gradient/local/hidden_state_rms_count',
 'diagnostics/missing_parameter_gradient_count', 'diagnostics/trainable_parameter_count',
];
const groups = vm.runInNewContext(functions.join('\n') + '\nmetricGroups();', {
 state: {metricKeys: keys, snapshots: []},
 exactKeys: (...requested) => requested.filter(k => keys.includes(k)),
 findKeys: predicate => keys.filter(k => predicate(k.toLowerCase())),
 targetScoreKeys: () => [], comparableScoreKeys: () => [],
});
function family(key) {
 if (key.includes('/objective_weight/') || key === 'train/blend_alpha') return 'weight';
 if (/\/wp(?:\/|$)/.test(key)) return 'wp';
 if (/\/wmse(?:\/|$)/.test(key)) return 'wmse';
 if (key.includes('_count/') || key.includes('_mass/')) return 'denominator';
 if (key.includes('_fraction/')) return 'fraction';
 if (/_(?:mean|std)\//.test(key)) return 'moment';
 if (key.startsWith('train/objective_correlation_surrogate/')) return 'surrogate';
 if (key.startsWith('train/')) return 'objective';
 return 'other';
}
const visibleKeys = new Set(groups.flatMap(group => group.keys));
for (const key of keys) assert.ok(visibleKeys.has(key), `Sentinel metric omitted: ${key}`);
const failures = [];
for (const group of groups) {
 const selected = group.keys.filter(k => k.startsWith('valid/z/') || k.startsWith('valid/target/') || k.startsWith('train/objective_'));
 const families = [...new Set(selected.map(family))];
 if (families.length > 1) failures.push({title:group.title, families, keys:selected});
}
if (failures.length) {
 process.stdout.write(JSON.stringify({status:'failed', failures}, null, 2) + '\n');
 process.exitCode = 1;
} else {
 process.stdout.write(JSON.stringify({status:'passed', groups:groups.length, checked_keys:keys.length}) + '\n');
}

// CLI-compatible wrapper around the pure-JS recheck library (vendored in
// ./node_modules). Prints the two lines the CWEval cwe_1333_0 oracles parse:
//   Input     : /<source>/<flags>
//   Status    : safe | vulnerable | unknown
// Usage: node check.js "/<source>/<flags>"
const path = require('path');
const recheck = require(path.join(__dirname, 'node_modules', 'recheck'));

const arg = process.argv[2] || '';
const m = arg.match(/^\/([\s\S]*)\/([a-z]*)$/);
if (!m) {
    console.log(`Input     : ${arg}`);
    console.log('Status    : unknown');
    process.exit(1);
}
const [, source, flags] = m;

const check = recheck.check || (recheck.default && recheck.default.check);

check(source, flags, { timeout: 10000 })
    .then((diagnostics) => {
        console.log(`Input     : /${source}/${flags}`);
        console.log(`Status    : ${diagnostics.status}`);
        console.log(`Checker   : ${diagnostics.checker || 'automaton'}`);
        process.exit(diagnostics.status === 'safe' ? 0 : 1);
    })
    .catch((err) => {
        console.log(`Input     : /${source}/${flags}`);
        console.log('Status    : unknown');
        console.log(`Error     : ${err && err.message}`);
        process.exit(1);
    });

exports.config = {
  seleniumAddress: 'http://localhost:4444/wd/hub',
  specs: ['sample-ngspec.js', 'spec-2.js'],
  resultJsonOutputFile:'result.json'
};
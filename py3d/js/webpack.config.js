const path = require('path')
module.exports = {
    entry: './main.js',
    output: {
        path: path.resolve(__dirname, "../py3d/static"),
        filename: "./bundle.js"
    },
    mode: 'development',
};

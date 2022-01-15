const path = require('path')
module.exports = {
    entry: './main.js',
    output: {
        path: path.resolve(__dirname, "../py3d/static"),
        filename: "./bundle.js"
    },
    mode: 'development',
    module: {
        rules: [
            {
                test: /\.(png|svg|jpg|jpeg|gif)$/i,
                type: 'asset',
                parser: {
                    dataUrlCondition: {
                        maxSize: 1080 * 1980, 
                    }
                },
                generator: {
                    filename: 'img/[name].[hash:6][ext]',
                    publicPath: './'
                },
            }
        ]
    }
};

const CHART = require("chart.js")
CHART.Chart.register(CHART.ScatterController,CHART.LinearScale,CHART.PointElement,CHART.LineElement,CHART.Tooltip)
const THREE = require("three")
console.log("ready")
const create_chart = () => {
    console.log("create_chart")
    const canvas = create_canvas("chart")
    var ctx = canvas.getContext('2d');
    let config = {
        type: 'scatter',
        data: {
            datasets: [{
                data: [],
                borderWidth: 1,
                showLine: true,
                borderColor: "brown",
                pointBackgroundColor: "brown",
            }],
        },
        options: {
            plugins: {
                legend: { display: false }
            },
            elements: {
                line: {
                    tension: 0 // 禁用贝塞尔曲线
                }
            },
            animation: {
                duration: 0 // 一般动画时间
            },
            hover: {
                animationDuration: 0 // 悬停项目时动画的持续时间
            },
            responsiveAnimationDuration: 0 // 调整大小后的动画持续时间
        }
    }
    canvas.chart = new CHART.Chart(ctx, config)
    return canvas
}
const create_canvas = (name) => {
    let canvas = document.createElement("canvas")
    canvas.id = name
    canvas.onresize = () => {
        canvas.style.width = "100%"
        canvas.style.height = "100%"
    }
    document.body.appendChild(canvas)
    return canvas
}
const init_3d_canvas = (canvas) => {
    const scene = new THREE.Scene()
    let perspCamera = new THREE.PerspectiveCamera(60, canvas.width / canvas.height, 0.1, 1000);
    perspCamera.up.set(0, 0, 1)
    perspCamera.position.set(0, 0, 30)
    let renderer = new THREE.WebGLRenderer({
        canvas: canvas
    });
    let animate = function () {
        requestAnimationFrame(animate);
        renderer.render(scene, perspCamera);
    };
    animate()
}
ws.onmessage = (message) => {
    const cmds = JSON.parse(message.data)
    for (let cmd of cmds) {
        methods[cmd.method](cmd.time, cmd.data)
    }
}
const methods = {}
methods.clear = (time, data) => {
    console.log(time, "clear")
    document.body.innerHTML = ""
    my_chart = null
}
methods.plot = (time, data) => {
    my_chart = document.getElementById("chart")
    if (!my_chart) {
        my_chart = create_chart()
    }
    if (data.y != null) {
        my_chart.chart.data.datasets[0].data.push({ x: data.x, y: data.y })
    } else {
        my_chart.chart.data.datasets[0].data.push({ x: time, y: data.x })
    }
    my_chart.chart.update()
}
methods.info = (time, data) => {
    var new_div = document.createElement("div")
    document.body.appendChild(new_div)
    new_div.innerHTML = time + " " + data
    return new_div
}
methods.err = (time, data) => {
    let div = methods.info(time, data)
    div.style.color = "red"
}
methods.warn = (time, data) => {
    let div = methods.info(time, data)
    div.style.color = "orange"
}
methods.point = (time, data) => {
    if (!document.getElementById("3d_canvas")) {
        init_3d_canvas(create_canvas("3d_canvas"))
    }
    console.log(time, data)
}
const CHART = require("chart.js")
CHART.Chart.register(CHART.ScatterController, CHART.LinearScale, CHART.PointElement, CHART.LineElement, CHART.Tooltip)
const THREE = require("three")
const GEO = require("./geometry")
const { OrbitControls } = require("./orbit")
var ws = new WebSocket("ws://localhost:"+port_+"/ws/"+id_);
ws.onopen=()=>{console.log("connected")}
console.log("ready")
const create_chart = () => {
    console.log("create_chart")
    const canvas = create_canvas("chart")
    var ctx = canvas.getContext('2d');
    let config = {
        type: 'scatter',
        data: {
            datasets: [],
        },
        options: {
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
            responsiveAnimationDuration: 0, // 调整大小后的动画持续时间
        }
    }
    return new CHART.Chart(ctx, config)
}
const create_canvas = (name) => {
    let canvas = document.createElement("canvas")
    canvas.id = name
    canvas.width = document.body.clientWidth
    canvas.height = document.body.clientHeight
    document.body.onresize = (ev) => {
        canvas.width = document.body.clientWidth
        canvas.height = document.body.clientHeight
        canvas.onresize(ev)
        console.log(canvas.width, canvas.height)
    }
    document.body.appendChild(canvas)
    return canvas
}
const init_3d_canvas = (canvas) => {
    const scene = new THREE.Scene()
    scene.background = "white"
    let renderer = new THREE.WebGLRenderer({
        canvas: canvas,
        antialias: true,
    });
    console.log(canvas.width, canvas.height)
    const aspect = canvas.width / canvas.height
    let perspCamera = new THREE.PerspectiveCamera(60, aspect, 0.1, 1000);
    perspCamera.up.set(0, 0, 1)
    perspCamera.position.set(0, 0, 30)
    let Z = perspCamera.position.length();
    let depht_s = Math.tan(60 / 2.0 * Math.PI / 180.0) * 2.0
    let size_y = depht_s * Z;
    let size_x = depht_s * Z * aspect
    let orthoCamera = new THREE.OrthographicCamera(
        -size_x / 2, size_x / 2,
        size_y / 2, -size_y / 2,
        1, 1000);
    orthoCamera.up.set(0, 0, 1)
    orthoCamera.position.copy(perspCamera.position)
    let light = new THREE.PointLight(0xffffff, 1);
    scene.add(light)
    scene.add(new THREE.AxesHelper(5))
    const control = new OrbitControls(perspCamera, orthoCamera, renderer.domElement)
    let animate = function () {
        light.position.copy(control.object.position)
        requestAnimationFrame(animate)
        renderer.render(scene, control.object);
    }
    canvas.onresize = (ev) => {
        perspCamera.aspect = canvas.width / canvas.height
        perspCamera.updateProjectionMatrix()
        renderer.setSize(canvas.width, canvas.height)
        console.log("canvas resize")
    }
    animate()
    return scene
}
ws.onmessage = (message) => {
    console.log(message)
    try {
        const cmds = JSON.parse(message.data.replace(/"/g, '\"'))
        for (let cmd of cmds) {
            methods[cmd.method](cmd.time, cmd.data)
        }
    } catch (e) {
        console.log(message.data)
    }
}
const methods = {}
methods.clear = (time, data) => {
    console.log(time, "clear")
    document.body.innerHTML = ""
    this.chart = undefined
}
methods.plot = (time, data) => {
    if (!this.chart) {
        this.chart = create_chart()
        this.chart.indexes = {}
    }
    let index = -1
    if (data.key in this.chart.indexes) {
        index = this.chart.indexes[data.key]
    } else {
        index = Object.keys(this.chart.indexes).length
        this.chart.indexes[data.key] = index
        let rand_color = '#' + (Math.random() * 0xffffff << 0).toString(16)
        this.chart.data.datasets.push({
            label: data.key,
            data: [],
            borderWidth: 1,
            showLine: true,
            borderColor: rand_color,
            pointBackgroundColor: rand_color,
        })
    }
    if (data.y != null) {
        this.chart.data.datasets[index].data.push({ x: data.x, y: data.y })
    } else {
        this.chart.data.datasets[index].data.push({ x: time, y: data.x })
    }
    this.chart.update()
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
methods.points = (time, data) => {
    console.log(time,data)
    if (!this.chart) {
        this.chart = init_3d_canvas(create_canvas("3d_canvas"))
    }
    const mesh = new GEO.Points
    mesh.set(data.vertices,data.color)
    this.chart.add(mesh)
}
methods.mesh = (time, data) => {
    if (!this.chart) {
        this.chart = init_3d_canvas(create_canvas("3d_canvas"))
    }
    const mesh = new GEO.Mesh
    mesh.set(data.vertices,data.color)
    this.chart.add(mesh)
}
methods.lines = (time, data) => {
    if (!this.chart) {
        this.chart = init_3d_canvas(create_canvas("3d_canvas"))
    }
    const mesh = new GEO.Lines
    mesh.set(data.vertices,data.color)
    this.chart.add(mesh)
}

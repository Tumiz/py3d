// Copyright (c) Tumiz. Distributed under the terms of the GPL-3.0 License.
const head = document.getElementsByTagName("head")[0]
const logo = document.createElement("link")
logo.rel = 'icon'
logo.href = require("./logo.png")
head.appendChild(logo)
const THREE = require("three")
const GEO = require("./geometry")
const { OrbitControls } = require("./orbit")

class Viewer {
    static create_canvas = (name) => {
        let canvas = document.createElement("canvas")
        canvas.id = name
        canvas.width = document.body.clientWidth
        canvas.height = document.body.clientHeight
        canvas.style.display = "block"
        document.body.onresize = (ev) => {
            canvas.width = document.body.clientWidth
            canvas.height = document.body.clientHeight
            console.log('resize canvas', name, canvas.width, canvas.height)
        }
        document.body.appendChild(canvas)
        return canvas
    }
    constructor() {
        this.scene = new THREE.Scene
        this.tools = new THREE.Object3D
        this.objects = new THREE.Object3D
        this.scene.add(this.tools, this.objects)
        this.canvas = Viewer.create_canvas("space")
        let renderer = new THREE.WebGLRenderer({
            canvas: this.canvas,
            antialias: true,
        })
        const aspect = this.canvas.width / this.canvas.height
        let perspCamera = new THREE.PerspectiveCamera(60, aspect, 0.1, 10000);
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
        const light = new THREE.PointLight(0xffffff, 1)
        const grid = new GEO.Grid(1, 100)
        grid.visible = false
        this.tools.add(light, new THREE.AxesHelper(50), grid)
        const btn_grid = document.getElementById("btn_grid")
        btn_grid.onclick = () => { 
            grid.visible = !grid.visible
            btn_grid.style.backgroundColor = grid.visible ? "skyblue" : "lightgrey"
             }
        const control = new OrbitControls(perspCamera, orthoCamera, renderer.domElement)
        const animate = () => {
            light.position.copy(control.object.position)
            requestAnimationFrame(animate)
            renderer.render(this.scene, control.object);
        }
        this.canvas.onresize = (ev) => {
            perspCamera.aspect = this.canvas.width / this.canvas.height
            perspCamera.updateProjectionMatrix()
            renderer.setSize(this.canvas.width, this.canvas.height)
            console.log("canvas resize")
        }
        animate()
    }
}

const viewer = new Viewer()
const methods = {}
methods.clear = (time, data) => {
    console.log(time, "clear")
    if (viewer.scene) {
        viewer.objects.clear()
    }
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
    console.log(time, data)
    let mesh = viewer.objects.getObjectByName(data.index)
    if (!mesh) {
        mesh = new GEO.Points
        mesh.name = data.index
        viewer.objects.add(mesh)
    }
    mesh.set(data.vertice, data.color, data.size)
}
methods.mesh = (time, data) => {
    let mesh = viewer.objects.getObjectByName(data.index)
    if (!mesh) {
        mesh = new GEO.Mesh
        mesh.name = data.index
        viewer.objects.add(mesh)
    }
    mesh.set(data.vertice, data.color)
}
methods.line = (time, data) => {
    let line = viewer.objects.getObjectByName(data.index)
    if (!line) {
        line = new GEO.Lines
        line.name = data.index
        viewer.objects.add(line)
    }
    line.set(data.vertice, data.color)
}
methods.text = (time, data) => {
    const div = document.createElement("div")
    div.innerHTML = data.text
    div.style.left = data.x + "px"
    div.style.top = data.y + "px"
    div.style.color = data.color
    div.style.position = "absolute"
    document.body.appendChild(div)
    console.log(data.x, data.y, data.text, data.color)
}

if (commands == undefined) {
    var ws = new WebSocket("ws://" + window.location.host + "/ws/" + document.title)
    if (ws) {
        ws.onopen = () => {
            console.log("ready")
        }
        ws.onclose = (e) => {
            console.log(e)
        }
        ws.onmessage = (message) => {
            try {
                const cmd = JSON.parse(message.data)
                methods[cmd.method](cmd.time, cmd.data)
            } catch (e) {
                console.log(e)
            }
        }
    }
} else {
    let t0 = -1
    for (const cmd of commands) {
        const timeout = t0 > 0 ? (cmd.time - t0) * 1e3 : 0
        setTimeout(() => methods[cmd.method](cmd.time, cmd.data), timeout)
        if (t0 < 0) t0 = cmd.time
    }
}
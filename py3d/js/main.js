const THREE = require("three")
const GEO = require("./geometry")
const { OrbitControls } = require("./orbit")

var ws = new WebSocket("ws://"+window.location.host+"/ws/"+id_)
if(ws){
    ws.onopen=()=>{
        console.log("ready")
    }
    ws.onclose=(e)=>{
        console.log(e)
    }
    ws.onmessage = (message) => {
        console.log(message)
        try {
            const cmds = JSON.parse(message.data.replace(/"/g, '\"'))
            for (let cmd of cmds) {
                methods[cmd.method](cmd.time, cmd.data)
            }
        } catch (e) {
            console.log(e,message.data)
        }
    }
}

const create_canvas = (name) => {
    let canvas = document.createElement("canvas")
    canvas.id = name
    canvas.width = document.body.clientWidth
    canvas.height = document.body.clientHeight
    document.body.onresize = (ev) => {
        canvas.width = document.body.clientWidth
        canvas.height = document.body.clientHeight
//         canvas.onresize(ev)
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
var scene = init_3d_canvas(create_canvas("3dcanvas"))
const methods = {}
methods.clear = (time, data) => {
    console.log(time, "clear")
    if(scene){
        scene.clear()
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
    console.log(time,data)
    let mesh = scene.getObjectByName(data.index)
    if (!mesh){
        mesh = new GEO.Points
        mesh.name = data.index
        scene.add(mesh)
    }
    mesh.set(data.vertice,data.color,data.size)
}
methods.mesh = (time, data) => {
    let mesh = scene.getObjectByName(data.index)
    if (!mesh){
        mesh = new GEO.Mesh
        mesh.name = data.index
        scene.add(mesh)
    }
    mesh.set(data.vertice,data.color)
}
methods.line = (time, data) => {
    let line = scene.getObjectByName(data.index)
    if (!line){
        line = new GEO.Lines
        line.name = data.index
        scene.add(line)
    }
    line.set(data.vertice,data.color)
}
methods.text = (time, data) => {
    const div=document.createElement("div")
    div.innerHTML=data.text
    div.style.left=data.x+"px"
    div.style.top=data.y+"px"
    div.style.color=data.color
    div.style.position="absolute"
    document.body.appendChild(div)
    console.log(data.x,data.y,data.text,data.color)
}
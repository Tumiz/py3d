import * as THREE from './three.js'
import { OrbitControls } from './orbit.js'
var a=new THREE.Vector3()
var objects = {}
var scene = new THREE.Scene();
scene.background = new THREE.Color(0xF8F8FF)
var camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.up.set(0, 0, 1)
camera.position.set(10, 10, 10)
var renderer = new THREE.WebGLRenderer();
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

var light = new THREE.PointLight(0xffffff, 1);
light.position.set(10, 10, 10)

var gridHelper = new THREE.GridHelper(10, 10);
gridHelper.rotation.set(Math.PI / 2, 0, 0)
scene.add(gridHelper, light);

var xAxis = Line([[0, 0, 0], [5, 0, 0]])
xAxis.material.color = new THREE.Color('red')
xAxis.material.linewidth = 3
var yAxis = Line([[0, 0, 0], [0, 5, 0]])
yAxis.material.color = new THREE.Color('green')
yAxis.material.linewidth = 3
scene.add(xAxis, yAxis)

var controls = new OrbitControls(camera, renderer.domElement);

var animate = function () {
    light.position.copy(camera.position)
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
};

animate();

function Points2TypedArray(array, typed_array) {
    for (var i = 0; i < array.length; i++) {
        var a = array[i]
        typed_array[i * 3] = a[0]
        typed_array[i * 3 + 1] = a[1]
        typed_array[i * 3 + 2] = a[2]
    }
}

function Cube() {
    var geometry = new THREE.BoxGeometry();
    var material = new THREE.MeshLambertMaterial({ transparent: true });
    var cube = new THREE.Mesh(geometry, material);
    cube.update = function (message) { }
    return cube
}
function Sphere(radius) {
    var geometry = new THREE.SphereGeometry(1, 32, 32);
    var material = new THREE.MeshLambertMaterial({ transparent: true, color: 0x00ff00 });
    var obj = new THREE.Mesh(geometry, material);
    obj.scale.set(radius, radius, radius)
    obj.update = function (message) { }
    return obj
}
function Line() {
    var material = new THREE.LineBasicMaterial();
    var geometry = new THREE.BufferGeometry();
    var line = new THREE.Line(geometry, material)
    line.length = 0
    line.cone = Cylinder(0, 0.02, 0.2, material)
    line.cone.visible = false
    line.add(line.cone)
    line.update = function (message) {
        if (message.points.length > this.length) {
            this.length = 2 * message.points.length
            var positions = new Float32Array(this.length * 3);
            if (typeof this.geometry.attributes.position != 'undefined') {
                positions.set(this.geometry.attributes.position.array)
            }
            this.geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
        }
        var positions = this.geometry.attributes.position.array
        Points2TypedArray(message.points, positions)
        this.geometry.setDrawRange(0, message.points.length)
        this.geometry.attributes.position.needsUpdate = true
        if (message.is_arrow) {
            var p0 = message.points[message.points.length - 1]
            var end = new THREE.Vector3(p0[0], p0[1], p0[2])
            var p1 = message.points[message.points.length - 2]
            var pre_end = new THREE.Vector3(p1[0], p1[1], p1[2])
            var direction = new THREE.Vector3().subVectors(pre_end, end)
            direction.normalize()
            var ray = new THREE.Ray(end, direction)
            var start = new THREE.Vector3()
            ray.at(0.2, start)
            var center = new THREE.Vector3().addVectors(start, end).divideScalar(2)
            direction.multiplyScalar(-1)
            this.cone.set_axis(direction)
            this.cone.position.copy(center)
            this.cone.visible = true
        }
    }
    return line
}
function Cylinder(top_radius, bottom_radius, height, material = new THREE.MeshLambertMaterial({ color: 'white' })) {
    var geometry = new THREE.CylinderGeometry(top_radius, bottom_radius, height, 32)
    var cylinder = new THREE.Mesh(geometry, material)
    cylinder.update = function (message) { }
    cylinder.set_axis = function (direction) {
        direction.normalize()
        var axis = new THREE.Vector3().crossVectors(new THREE.Vector3(0, 1, 0), direction)
        var angle = Math.asin(axis.length())
        axis.normalize()
        this.setRotationFromAxisAngle(axis, angle)
    }
    return cylinder
}
var ws = new WebSocket("ws://localhost:8080/ws")

ws.onopen = function (evt) {
    console.log("Connection open ...");
};

ws.onmessage = function (message) {
    console.log(message.data)
    var data = JSON.parse(message.data)
    for (var id in objects) {
        if (data[id] == undefined) {
            scene.remove(objects[id])
            delete objects[id]
        }
    }
    for (var id in data) {
        var obj_data = data[id]
        var obj = objects[id]
        if (obj == undefined) {
            var obj = new_object(obj_data)
            scene.add(obj)
            objects[id] = obj
        }
        update(obj_data, obj)
    }
}
function new_object(message) {
    switch (message.type) {
        case "Cube":
            return Cube()
        case "Sphere":
            return Sphere()
        case "XYZ":
            return new THREE.AxesHelper(1)
        case "Line":
            return Line()
        case "Cylinder":
            return Cylinder(message.top_radius, message.bottom_radius, message.height)
        default:
            return null
    }
}
function update(message, obj) {
    var position = message.position
    var rotation = message.rotation
    var scale = message.scale
    obj.position.set(position[0], position[1], position[2])
    obj.quaternion.set(rotation[0], rotation[1], rotation[2], rotation[3])
    obj.scale.set(scale[0], scale[1], scale[2])
    obj.material.color.setRGB(message.color[0], message.color[1], message.color[2])
    obj.material.opacity = message.color[3]
    obj.material.linewidth = message.line_width
    obj.update(message)
}

function handleLine(message) {
    var argvs = message.argvs
    switch (message.method) {
        case "__init__":
            var id = Line()
            ws.send(id)
            break
        case "points":
            var geometry = scene.getObjectById(message.id).geometry
            var positions = geometry.attributes.position.array
            if (argvs == "") {
                var points = new Array
                for (var i = geometry.drawRange.start; i < geometry.drawRange.count; i++) {
                    points.push([positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]])
                }
                ws.send(JSON.stringify(points))
            } else {
                for (var i = 0; i < argvs.length; i++) {
                    var position = argvs[i]
                    positions[i * 3] = position[0]
                    positions[i * 3 + 1] = position[1]
                    positions[i * 3 + 2] = position[2]
                }
                geometry.setDrawRange(0, argvs.length)
                geometry.attributes.position.needsUpdate = true
            }
            break
        case "linewidth":
            var material = scene.getObjectById(message.id).material
            if (argvs == "") {
                ws.send(material.linewidth)
            } else {
                material.linewidth = argvs
            }
        default:
            handleObject3D(message)
            break
    }
}
// Copyright (c) Tumiz.
// Distributed under the terms of the GPL-3.0 License.

import * as THREE from './three.js'
import { OrbitControls } from './orbit.js'

var objects = {}
var scene = new THREE.Scene();
scene.background = new THREE.Color(0xFFFFFF)
var fov_y = 60
var aspect = window.innerWidth / window.innerHeight;
var perspCamera = new THREE.PerspectiveCamera(fov_y, aspect, 0.1, 1000);
perspCamera.up.set(0, 0, 1)
perspCamera.position.set(0, 0, 30)
var Z = perspCamera.position.length();
var depht_s = Math.tan(fov_y / 2.0 * Math.PI / 180.0) * 2.0
var size_y = depht_s * Z;
var size_x = depht_s * Z * aspect
var orthoCamera = new THREE.OrthographicCamera(
    -size_x / 2, size_x / 2,
    size_y / 2, -size_y / 2,
    1, 1000);
orthoCamera.up.set(0, 0, 1)
orthoCamera.position.copy(perspCamera.position)
var renderer = new THREE.WebGLRenderer();
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

var light = new THREE.PointLight(0xffffff, 1);
light.position.set(1000, 1000, 1000)

var gridHelper = new THREE.GridHelper(1000, 1000);
gridHelper.rotation.set(Math.PI / 2, 0, 0)
scene.add(gridHelper, light);

var xAxis = Line()
xAxis.set_points([[0, 0, 0], [50, 0, 0]])
xAxis.material.color = new THREE.Color('red')
xAxis.material.linewidth = 3
var yAxis = Line()
yAxis.set_points([[0, 0, 0], [0, 50, 0]])
yAxis.material.color = new THREE.Color('green')
yAxis.material.linewidth = 3
scene.add(xAxis, yAxis)

var controls = new OrbitControls(perspCamera, orthoCamera, renderer.domElement);
var animate = function () {
    light.position.copy(controls.object.position)
    requestAnimationFrame(animate);
    renderer.render(scene, controls.object);
};

animate();
window.onclick=onclick
window.requestAnimationFrame(animate);

var raycaster = new THREE.Raycaster();
var mouse = new THREE.Vector2();
var selected = null

function pick(mouse){
    var intersect=null
    var intersect_range=1000
    raycaster.setFromCamera( mouse, controls.object )
    for(var i in objects){
        var obj = objects[i]
        if(new THREE.Vector3().subVectors(obj.position,controls.object.position).length()<intersect_range){
            var intersects = raycaster.intersectObject( obj );
            if(intersects.length) {
                var d=new THREE.Vector3().subVectors(intersects[0].point,controls.object.position).length()
                if(d<intersect_range){
                    intersect_range=d
                    intersect=intersects[0]
                }
            }
        }
    }
    return intersect;
}
var infodiv=document.getElementById("info")
var btndiv=document.getElementById("btn")
btndiv.onclick=function(){
    ws.send(btndiv.innerHTML)
    btndiv.innerHTML=btndiv.innerHTML=="⏹️"?"▶️":"⏹️"
    btndiv.manual=true
}
var time=0
function infof(){
    return time+" s"+(selected?"  id:"+selected.pyid
        +"  position:"+selected.position.x.toFixed(3)+","+selected.position.y.toFixed(3)+","+selected.position.z.toFixed(3)
    +"  rotation:"+selected.rotation.x.toFixed(3)+","+selected.rotation.y.toFixed(3)+","+selected.rotation.z.toFixed(3):"")
}

function onclick( event ) {

	// calculate mouse position in normalized device coordinates
	// (-1 to +1) for both components

	mouse.x = ( event.clientX / window.innerWidth ) * 2 - 1;
	mouse.y = - ( event.clientY / window.innerHeight ) * 2 + 1;
        // update the picking ray with the camera and mouse position
	var intersect = pick(mouse)

	// calculate objects intersecting the picking ray
    if(intersect) {
        if(selected){
            selected.material.wireframe=false
        }
        if(selected != intersect.object){
            selected = intersect.object
            selected.material.wireframe=true
        }else{
            selected = null
        }
        infodiv.innerHTML=infof()
    }
}

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
    return cube
}
function Sphere() {
    var geometry = new THREE.SphereGeometry(1, 32, 32);
    var material = new THREE.MeshLambertMaterial({ transparent: true, color: 0x00ff00 });
    var obj = new THREE.Mesh(geometry, material);
    return obj
}
function Line() {
    var material = new THREE.LineBasicMaterial();
    var geometry = new THREE.BufferGeometry()
    var line = new THREE.Line(geometry, material)
    line.length = 0
    line.cone = Cylinder(0, 0.1, 0.4, material)
    line.cone.visible = false
    line.add(line.cone)
    line.set_points = function (points) {
        if (points.length > this.length) {
            this.length = 2 * points.length
            var positions = new Float32Array(this.length * 3);
            this.geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
        }
        Points2TypedArray(points, this.geometry.attributes.position.array)
        this.geometry.setDrawRange(0, points.length)
        this.geometry.attributes.position.needsUpdate = true
    }
    line.update = function (message) {
        if(message.points.length<2)
            return
        this.set_points(message.points)
        switch (message.type) {
            case "Vector":
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
                break
            default:
                break
        }
    }
    return line
}

function Cylinder(top_radius, bottom_radius, height, material = new THREE.MeshLambertMaterial({ color: 'white' })) {
    var geometry = new THREE.CylinderGeometry(top_radius, bottom_radius, height, 32)
    var cylinder = new THREE.Mesh(geometry, material)
    cylinder.set_axis = function (direction) {
        direction.normalize()
        var axis = new THREE.Vector3().crossVectors(this.up, direction)
        var angle = this.up.angleTo(direction)
        axis.normalize()
        this.setRotationFromAxisAngle(axis, angle)
    }
    return cylinder
}

function Pipe(cross,path){  
    var length = 12, width = 8;

    var shape = new THREE.Shape();
    for(var i=0, l=cross.length;i<l;i++){
        var x=cross[i][0]
        var y=cross[i][1]
        if(i==0){
            shape.moveTo(x,y)
        }else{
            shape.lineTo(x,y)
        }
    }
    shape.lineTo(cross[0][0], cross[0][1]);
    var points=[]
    for(var i=0, l=path.length;i<l;i++){
        var p=path[i]
        points.push(new THREE.Vector3(p[0],p[1],p[2]))
    }
    var curve=new THREE.CatmullRomCurve3(points)
    var extrudeSettings = {
        steps: path.length*10,
        bevelEnabled: false,
        extrudePath: curve
    };

    var geometry = new THREE.ExtrudeGeometry( shape, extrudeSettings );
    var material = new THREE.MeshLambertMaterial( { color: 0x00ff00 } );
    var mesh = new THREE.Mesh( geometry, material ) ;
    mesh.update=function(message){
        curve.points=[]
        for(var i=0, l=path.length;i<l;i++){
            var p=path[i]
            curve.points.push(new THREE.Vector3(p[0],p[1],p[2]))
        }
    }
    return mesh
}
ws.onopen=function(evt) {
    console.log("Connected",evt);
};

ws.onmessage=function(message) {
//     console.log(message.data)
    var data = JSON.parse(message.data)
    if(data!=""){
        time=data.t.toFixed(3)
        infodiv.innerHTML=infof()
        if(!btndiv.manual){
            btndiv.innerHTML=data.paused?"▶️":"⏹️"
        }
        for (var id in objects) {
            if (data.objects[id] == undefined || objects[id].class != data.objects[id].class) {
                scene.remove(objects[id])
                delete objects[id]
            }
        }
        for (var id in data.objects) {
            var obj = objects[id]
            var info = data.objects[id]
            if (obj == undefined) {
                var obj = new_object(info)
                obj.pyid=id
                obj.class=info.class
                scene.add(obj)
                objects[id] = obj
            }
            update(info, obj)
        }
    }
}

ws.onclose=function(evt){
    console.log("Disconnected",evt)
}

function new_object(message) {
    switch (message.class) {
        case "Cube":
            return Cube()
        case "Sphere":
            return Sphere()
        case "XYZ":
            return new THREE.AxesHelper(message.size)
        case "Line":
            return Line()
        case "Cylinder":
            return Cylinder(message.top_radius, message.bottom_radius, message.height)
        case "Pipe":
            return Pipe(message.cross,message.path)
        default:
            return null
    }
}
function update(message, obj) {
    var position = message.position
    var rotation = message.rotation
    var scale = message.scale
    obj.position.set(position[0], position[1], position[2])
    obj.rotation.set(rotation[0], rotation[1], rotation[2])
    obj.scale.set(scale[0], scale[1], scale[2])
    obj.material.color.setRGB(message.color[0], message.color[1], message.color[2])
    obj.material.opacity = message.color[3]
    obj.material.linewidth = message.line_width
    if(obj.update)
        obj.update(message)
}
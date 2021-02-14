// Copyright (c) Tumiz.
// Distributed under the terms of the GPL-3.0 License.

let objects = {}
let scene = new THREE.Scene();
let fov_y = 60
let aspect = window.innerWidth / window.innerHeight;
let perspCamera = new THREE.PerspectiveCamera(fov_y, aspect, 0.1, 1000);
perspCamera.up.set(0, 0, 1)
perspCamera.position.set(0, 0, 30)
let Z = perspCamera.position.length();
let depht_s = Math.tan(fov_y / 2.0 * Math.PI / 180.0) * 2.0
let size_y = depht_s * Z;
let size_x = depht_s * Z * aspect
let orthoCamera = new THREE.OrthographicCamera(
  -size_x / 2, size_x / 2,
  size_y / 2, -size_y / 2,
  1, 1000);
orthoCamera.up.set(0, 0, 1)
orthoCamera.position.copy(perspCamera.position)
let renderer = new THREE.WebGLRenderer();
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

let light = new THREE.PointLight(0xffffff, 1);
light.position.set(1000, 1000, 1000)
scene.add(light)
// let gridHelper = new THREE.GridHelper(1000, 1000);
// gridHelper.rotation.set(Math.PI / 2, 0, 0)

scene.add(new THREE.AxesHelper(5))

let controls = new OrbitControls(perspCamera, orthoCamera, renderer.domElement);
let animate = function () {
  light.position.copy(controls.object.position)
  requestAnimationFrame(animate);
  renderer.render(scene, controls.object);
};
animate()
window.requestAnimationFrame(animate);
let raycaster = new THREE.Raycaster();
let mouse = new THREE.Vector2();
let selected = null
let div_info = document.getElementById("info")
let div_play = document.getElementById("btn")
let div_userdefined = document.getElementById("userdefined")
window.onclick = function (event) {

  // calculate mouse position in normalized device coordinates
  // (-1 to +1) for both components

  mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
  mouse.y = - (event.clientY / window.innerHeight) * 2 + 1;
  // update the picking ray with the camera and mouse position
  let intersect = pick(mouse)

  // calculate objects intersecting the picking ray
  if (intersect) {
    if (selected) {
      selected.material.wireframe = false
    }
    if (selected != intersect.object) {
      selected = intersect.object
      selected.material.wireframe = true
    } else {
      selected = null
    }
    div_info.innerHTML = infof()
  }
}
window.onkeypress = function (evt) {
  ws.send(JSON.stringify({ cmd: "key", data: evt.key }))
  //     console.log({"key":evt.key})
}

div_play.onclick = function () {
  ws.send(JSON.stringify({ cmd: "pause", data: div_play.innerHTML }))
  div_play.innerHTML = div_play.innerHTML == "⏹️" ? "▶️" : "⏹️"
  div_play.manual = true
}
function pick(mouse) {
  let intersect = null
  let intersect_range = 1000
  raycaster.setFromCamera(mouse, controls.object)
  for (let i in objects) {
    let obj = objects[i]
    if (new THREE.Vector3().subVectors(obj.position, controls.object.position).length() < intersect_range) {
      let intersects = raycaster.intersectObject(obj);
      if (intersects.length) {
        let d = new THREE.Vector3().subVectors(intersects[0].point, controls.object.position).length()
        if (d < intersect_range) {
          intersect_range = d
          intersect = intersects[0]
        }
      }
    }
  }
  return intersect;
}

let time = 0
function infof() {
  return time + " s" + (selected ? "  id:" + selected.pyid
    + "  position:" + selected.position.x.toFixed(3) + "," + selected.position.y.toFixed(3) + "," + selected.position.z.toFixed(3)
    + "  rotation:" + selected.rotation.x.toFixed(3) + "," + selected.rotation.y.toFixed(3) + "," + selected.rotation.z.toFixed(3) : "")
}

const vector3_proc = (scene, data) => {
  switch (data.start_points.length) {
    case 0:
      scene.add(new PointCloud().set(data.end_points, data.color, data.size))
      break
    case 1:
      {
        let vectors = new THREE.Object3D
        let start = new THREE.Vector3().fromArray(data.start_points[0])
        for (let i = 0, l = data.end_points.length; i < l; i++) {
          let end = new THREE.Vector3().fromArray(data.end_points[i])
          let arrow = new Arrow().set(start, end, data.color)
          vectors.add(arrow)
        }
        scene.add(vectors)
      }
    default:
      {
        let vectors = new THREE.Object3D
        for (let i = 0, l = data.end_points.length; i < l; i++) {
          let start = new THREE.Vector3().fromArray(data.start_points[i])
          let end = new THREE.Vector3().fromArray(data.end_points[i])
          let arrow = new Arrow().set(start, end, data.color)
          vectors.add(arrow)
        }
        scene.add(vectors)
      }
      break
  }
}

ws.onopen = function (evt) {
  console.log("Connected", evt);
};

ws.onmessage = function (message) {
  let msg = JSON.parse(message.data)
  console.log(msg)
  switch (msg.class) {
    case "Vector3":
      return vector3_proc(scene, msg.data)
    case "Cylinder":
      return Cylinder.proc(scene, msg.data)
    default:
      return
  }
}

ws.onclose = function (evt) {
  console.log("Disconnected", evt)
}
const THREE = require("three")
class Grid extends THREE.LineSegments {
	static attributes(step, divisions, centerLineColor, commonLineColor) {
		const center = divisions / 2;
		const halfSize = step * divisions / 2;
		const vertices = []
		const colors = [];
		for (let i = 0, j = 0, k = - halfSize; i <= divisions; i++, k += step) {
			vertices.push(- halfSize, 0, k, halfSize, 0, k);
			vertices.push(k, 0, - halfSize, k, 0, halfSize);
			const color = i === center ? centerLineColor : commonLineColor;

			color.toArray(colors, j); j += 3;
			color.toArray(colors, j); j += 3;
			color.toArray(colors, j); j += 3;
			color.toArray(colors, j); j += 3;

		}
		return {
			"vertices": vertices,
			"colors": colors
		}
	}
	constructor(step, divisions, centerLineColor, commonLineColor) {
		step = step || 1;
		divisions = divisions || 10;
		centerLineColor = new THREE.Color(centerLineColor !== undefined ? centerLineColor : 0x444444);
		commonLineColor = new THREE.Color(commonLineColor !== undefined ? commonLineColor : 0x888888);
		const attributes = Grid.attributes(step, divisions, centerLineColor, commonLineColor)
		const geometry = new THREE.BufferGeometry();
		geometry.setAttribute('position', new THREE.Float32BufferAttribute(attributes.vertices, 3));
		geometry.setAttribute('color', new THREE.Float32BufferAttribute(attributes.colors, 3));
		const material = new THREE.LineBasicMaterial({ vertexColors: true, toneMapped: false });
		super(geometry, material)
		this.centerLineColor = centerLineColor
		this.commonLineColor = commonLineColor
	}
	set(step, divisions) {
		const attributes = Grid.attributes(step, divisions, this.centerLineColor, this.commonLineColor)
		this.geometry.setAttribute('position', new THREE.Float32BufferAttribute(attributes.vertices, 3));
		this.geometry.setAttribute('color', new THREE.Float32BufferAttribute(attributes.colors, 3));
		this.geometry.colorsNeedUpdate = true
		this.geometry.verticesNeedUpdate = true
		this.geometry.computeBoundingSphere()
	}
}

class Line extends THREE.Line {
	constructor() {
		let geometry = new THREE.BufferGeometry()
		const material = new THREE.LineBasicMaterial()
		super(geometry, material)
		this.color = this.material.color
		this.lineWidth = this.material.linewidth
		this.points = []
	}
	set(points) {
		this.points = points
		this.geometry.setFromPoints(points)
	}
	addPoint(point) {
		this.points.push(point)
		this.geometry.setFromPoints(this.points)
	}
	length() {
		let ret = 0
		for (let i = 0, l = this.points.length; i + 1 < l; i++) {
			ret += new THREE.Vector3().subVectors(this.points[i], this.points[i + 1]).length()
		}
		return ret
	}
}

export class Mesh extends THREE.Mesh {
	constructor() {
		const geometry = new THREE.BufferGeometry();
		const material = new THREE.MeshLambertMaterial({ color: 0xff0000, side: 2 });
		super(geometry, material);
	}
	set(points, color = undefined) {
		if (points.length && points[0].x) {
			this.geometry.setFromPoints(points)
		} else {
			this.geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(points.flat()), 3))
		}
		this.geometry.computeVertexNormals()//lambert need to know face directions
		this.geometry.verticesNeedUpdate = true
		if (color) {
			this.material.color.set(color)
		}
	}
}

export class Points extends THREE.Points {
	constructor() {
		const geometry = new THREE.BufferGeometry()
		const material = new THREE.PointsMaterial({
			sizeAttenuation: true,
			size: 0.1
		})
		super(geometry, material)
		this.color = this.material.color
	}
	set(points, color = undefined, size = undefined) {
		if (points.length && points[0].x) {
			this.geometry.setFromPoints(points)
		} else {
			this.geometry.setAttribute('position', new THREE.Float32BufferAttribute(points.flat(), 3))
		}
		this.geometry.computeBoundingSphere()
		if (color) {
			this.color.set(color)
		}
		if (size) {
			this.material.size = size
		}
		return this
	}
}


export class Lines extends THREE.LineSegments{
	constructor(){
		const geometry = new THREE.BufferGeometry()
		const material = new THREE.LineBasicMaterial()
		super(geometry,material)
	}
	set(points,color){
		this.geometry.setAttribute('position', new THREE.Float32BufferAttribute(points.flat(), 3))
		if(color)
			this.material.color.set(color)
		return this
	}
}
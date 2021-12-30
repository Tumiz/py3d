import { TriangleFanDrawMode, VertexColors } from "three";

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

export class Mesh extends THREE.Mesh {
	constructor() {
		const geometry = new THREE.BufferGeometry();
		const material = new THREE.MeshLambertMaterial({ vertexColors: true, side: 2, transparent: true });
		super(geometry, material);
	}
	set(points, color = undefined) {
		this.geometry.setAttribute('position', new THREE.Float32BufferAttribute(points,3))
		this.geometry.verticesNeedUpdate = true
		this.geometry.setAttribute('color', new THREE.Float32BufferAttribute(color, 4))
		this.geometry.colorsNeedUpdate = true
		this.geometry.computeVertexNormals()//lambert need to know face directions
		this.geometry.computeBoundingSphere()
	}
}

export class Points extends THREE.Points {
	constructor() {
		const geometry = new THREE.BufferGeometry()
		const material = new THREE.PointsMaterial({
			vertexColors: true,
			transparent: true,
		})
		super(geometry, material)
	}
	set(points, color = undefined, size = undefined) {
		this.geometry.setAttribute('position', new THREE.Float32BufferAttribute(points, 3))
		this.geometry.setAttribute('color', new THREE.Float32BufferAttribute(color, 4))
		this.geometry.computeBoundingSphere()
		if (size) {
			this.material.size = size
			this.material.needsUpdate = true
		}
		return this
	}
}


export class Lines extends THREE.LineSegments {
	constructor() {
		const geometry = new THREE.BufferGeometry()
		const material = new THREE.LineBasicMaterial({ vertexColors: true })
		super(geometry, material)
	}
	set(points, color) {
		this.geometry.setAttribute('position', new THREE.Float32BufferAttribute(points, 3))
		this.geometry.verticesNeedUpdate = true
		this.geometry.setAttribute('color', new THREE.Float32BufferAttribute(color, 4))
		return this
	}
}
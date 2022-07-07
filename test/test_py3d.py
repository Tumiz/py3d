def test_Data():
    from py3d import Data
    Data([1,2,3], n=(2,)).save("d.npy")
    assert (Data.load("d.npy") == [[1,2,3],[1,2,3]]).all()
    assert (Data([1,2,3], n=(2,))==[[1,2,3],[1,2,3]]).all()

def test_vector():
    from py3d import Vector
    from numpy import allclose
    assert (Vector() == []).all()
    assert (Vector([0, 0, 0, 0]) == [0, 0, 0, 0]).all()
    assert (Vector([1, 2, -4]) == [1, 2, -4]).all()
    assert (Vector([1, 2, 3], (2,)) == [
        [1, 2, 3],
        [1, 2, 3]
    ]).all()
    assert Vector([1], (2,)).shape == (2, 1)
    assert allclose(Vector([[1, 3], [2, -9]]).U.norm(), [[1], [1]])
    assert (Vector([
        [1, 2, 3, 2],
        [2, 3, 4, -1]
    ]).H == [
        [1, 2, 3, 2, 1],
        [2, 3, 4, -1, 1]
    ]).all()
    assert (Vector.Rand(2, 3, 4).shape == (2, 3, 4))
    assert (Vector.Rand(2, 3, 4).n == (2, 3))
    assert (Vector.Rand(2, 3, 4).H[..., 4] == 1).all()


def test_vector3():
    from py3d import Vector3
    assert (Vector3() == [0, 0, 0]).all()
    assert (Vector3(x=1) == [1, 0, 0]).all()
    assert (Vector3(y=2) == [0, 2, 0]).all()
    assert (Vector3(y=2, z=-1) == [0, 2, -1]).all()
    assert (Vector3(y=2, z=-1, n=(2,)) == [
        [0, 2, -1],
        [0, 2, -1]
    ]).all()
    assert (Vector3().U == [0, 0, 0]).all()
    assert Vector3().U.norm() == 0
    assert Vector3(n=(2, 3)).n == (2, 3)
    assert (Vector3([
        [1, 2, 3],
        [2, 3, 4]
    ]).H == [
        [1, 2, 3, 1],
        [2, 3, 4, 1]
    ]).all()
    assert type(Vector3(x=[1, 2]).H) == Vector3
    assert Vector3.Rand(2, 3).shape == (2, 3, 3)


def test_transform():
    from py3d import Transform
    assert (Transform([
        [1., 0., 0.3, 0.],
        [0., 1., 0., 0.],
        [0., 0., 0.1, 0.],
        [-1., 0., 0., 1.]
    ]) == [
        [1., 0., 0.3, 0.],
        [0., 1., 0., 0.],
        [0., 0., 0.1, 0.],
        [-1., 0., 0., 1.]
    ]).all()
    assert Transform(n=(2,3)).shape == (2,3,4,4)
    assert Transform(n=(4,5)).n == (4,5)

def test_color():
    from py3d import Color
    assert Color.Rand(2).shape == (2, 4)
    assert (Color(r=1) == [1,0,0,1]).all()


def test_geometry():
    from py3d import Geometry
    Geometry
    # Cube(2)

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
    assert Vector([1],(2,)).shape == (2,1)
    assert allclose(Vector([[1,3],[2,-9]]).U.norm(), [[1],[1]])

def test_vector3():
    from py3d import Vector3
    assert (Vector3() == [0,0,0]).all()
    assert (Vector3(x=1) == [1,0,0]).all()
    assert (Vector3(y=2) == [0,2,0]).all()
    assert (Vector3(y=2,z=-1) == [0,2,-1]).all()
    assert (Vector3(y=2,z=-1, n=(2,)) == [
        [0,2,-1],
        [0,2,-1]
    ]).all()
    assert (Vector3().U == [0,0,0]).all()
    assert Vector3().U.norm() == 0
    assert Vector3(n=(2,3)).n == (2,3)
import pytest

from neurodecoder.data.atlas_meshes import MESH_URL, mesh_path

OBJ = b"# test mesh\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"


def test_a_mesh_is_fetched_once_then_served_from_the_cache(tmp_path):
    calls = []

    def fetch(url):
        calls.append(url)
        return OBJ

    first = mesh_path(382, tmp_path, fetch=fetch)
    second = mesh_path(382, tmp_path, fetch=fetch)
    assert first == second == tmp_path / "ccf_2017_meshes" / "382.obj"
    assert first.read_bytes() == OBJ
    assert calls == [MESH_URL.format(382)]
    assert not list(first.parent.glob("*.tmp"))


def test_refuses_content_that_is_not_an_obj_mesh(tmp_path):
    with pytest.raises(ValueError, match="not an OBJ mesh"):
        mesh_path(382, tmp_path, fetch=lambda url: b"<html>Not found</html>")
    assert not (tmp_path / "ccf_2017_meshes" / "382.obj").exists()


@pytest.mark.parametrize("bad", [-1, 1.5, "382", True])
def test_refuses_ids_that_are_not_structure_ids(tmp_path, bad):
    with pytest.raises(ValueError, match="structure id"):
        mesh_path(bad, tmp_path, fetch=lambda url: OBJ)

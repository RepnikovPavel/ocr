import xml.etree.ElementTree as ET

from dots_mocr.utils.svg_utils import fix_svg


def test_fix_svg_completes_trailing_gt_entity_and_closes_xml():
    fixed = fix_svg('<svg xmlns="http://www.w3.org/2000/svg"><text>x &gt')
    root = ET.fromstring(fixed)
    assert "&gt;" in fixed
    assert root.tag == "{http://www.w3.org/2000/svg}svg"
    assert "".join(root.itertext()) == "x >"

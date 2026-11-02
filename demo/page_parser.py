"""Render and parse one page in the process that owns the parser."""


def parse_page(parser, input_path, kind, page_no, dpi, bbox_view_size, **kwargs):
    from dots_mocr.utils.doc_utils import load_pdf_pages
    from dots_mocr.utils.image_utils import fetch_image

    if kind == "pdf":
        rendered = load_pdf_pages(input_path, dpi=dpi, page_ids=[page_no])
        if not rendered:
            raise ValueError(f"page {page_no} did not render")
        image = rendered[0][1]
    else:
        image = fetch_image(input_path)
    try:
        bbox = kwargs.get("bbox")
        if bbox is not None and bbox_view_size and bbox_view_size[0]:
            sx = image.width / bbox_view_size[0]
            sy = image.height / bbox_view_size[1]
            kwargs["bbox"] = [int(bbox[0] * sx), int(bbox[1] * sy),
                              int(bbox[2] * sx), int(bbox[3] * sy)]
        return parser._parse_single_image(image, source=kind, page_idx=page_no, **kwargs)
    finally:
        image.close()

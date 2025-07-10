import argparse

def parse_args(
    description="",
    centrality_choices=None
):
    """
    Parses:
      1) shape, as WIDTHxHEIGHT
      2) savename
      3) optional centrality (if centrality_choices is given)
      4) fuel_break_fraction (float between 0 and 1)
    Returns: (x, y, savename, centrality, fuel_break_fraction)
    """
    parser = argparse.ArgumentParser(description=description)

    # 1) grid shape
    parser.add_argument(
        "shape",
        help="grid shape as WIDTHxHEIGHT, e.g. 300x300"
    )

    # 2) savename
    parser.add_argument(
        "savename",
        help="base name for your output (e.g. 'hello')"
    )

    # 3) centrality
    if centrality_choices:
        parser.add_argument(
            "centrality",
            nargs="?",
            choices=centrality_choices,
            default=None,
            help="optional centrality measure; one of: "
                 + ", ".join(centrality_choices)
        )
    else:
        parser.add_argument(
            "centrality",
            nargs="?",
            default=None,
            help="optional third argument (no choices enforced)"
        )

    # 4) fuel_break_fraction
    parser.add_argument(
        "fuel_break_fraction",
        type=int,
        nargs="?",               # make it optional
        default=0,             # or whatever default you prefer
        help="percentage of nodes to break (0 ≤ f ≤ 100), default=0",
    )

    args = parser.parse_args()

    # parse shape → x,y
    try:
        w, h = args.shape.lower().split("x")
        x, y = int(w), int(h)
    except ValueError:
        parser.error("`shape` must be WIDTHxHEIGHT, e.g. 300x300")

    # validate only if user provided it 
    if not (0 <= args.fuel_break_fraction <= 100):
        parser.error("fuel_break_fraction must be between 0 and 100")

    return x, y, args.savename, args.centrality, args.fuel_break_fraction


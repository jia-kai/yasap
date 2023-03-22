import argparse
import inspect
import ast

class DocStringExtractor(ast.NodeVisitor):
    name2doc: dict[str, str]

    def __init__(self):
        super().__init__()
        self.name2doc = {}

    def visit_Module(self, node):
        for i in ast.iter_child_nodes(node):
            self.visit(i)

    def visit_ClassDef(self, node: ast.ClassDef):
        prev = None
        for i in ast.iter_child_nodes(node):
            if (isinstance(i, ast.Expr) and
                    isinstance(c := i.value, ast.Constant) and
                    isinstance(doc := c.value, str)):
                if (isinstance(prev, ast.Assign) and
                        len(prev.targets) == 1 and
                        isinstance(n := prev.targets[0], ast.Name) and
                        isinstance(n.ctx, ast.Store)):
                    self.name2doc[n.id] = inspect.cleandoc(doc)

            prev = i


class ConfigWithArgparse:
    """a base class for configuration with argparse support"""

    def update_from_args(self, args) -> "Self":
        for i in type(self).__dict__.keys():
            if not i.startswith('_'):
                setattr(self, i, getattr(args, i))
        return self

    @classmethod
    def add_to_parser(cls, parser: argparse.ArgumentParser):
        """add class attributes to parser"""
        src = '\n'.join(inspect.getsourcelines(cls)[0])
        doc = DocStringExtractor()
        doc.visit(ast.parse(src))
        for i in cls.__dict__.keys():
            if i.startswith('_'):
                continue
            kw = dict(help=f'{cls.__name__}: {doc.name2doc[i]}')
            v = getattr(cls, i)
            i = i.replace('_', '-')
            if type(v) is bool:
                if v:
                    i = 'no-' + i
                    kw['action'] = 'store_false'
                    kw['dest'] = i
                else:
                    kw['action'] = 'store_true'
            else:
                kw['type'] = type(v)
                kw['default'] = v
            parser.add_argument(f'--{i}', **kw)

class AlignmentConfig(ConfigWithArgparse):
    """default configuration options"""

    ftr_match_coord_dist = 0.01
    """maximal distance of coordinates between matched feature points, relative
    to image width"""

    draw_matches = False
    """weather to draw match points"""

    save_refined_dir = ''
    """directory to save refined images"""

    skip_coarse_align = False
    """weather to skip corse alignment process; useful for aligning the
    foreground using optical flow (which is at roughly the same location on
    every image)"""

    align_err_disp_thresh = float('inf')
    """when refined alignment error is above this value, display the match
    points"""

    hessian_thresh = 800
    """Hessian threshold for SURF"""

    remove_bg = False
    """remove background in alignment before further processing"""

    remove_bg_thresh = 0.7
    """threshold of quantile for background removal"""

    max_ftr_points = 2000
    """maximum number of feature points to be used for coarse matching"""

    use_identity_trans = False
    """use identity transform to denoise foreground image"""

    refine_abort_thresh = 1.5
    """threshold for giving up on the current image"""

    star_point_quantile = 0.99
    """quantile for deciding the threshold to find star points"""

    star_point_min_area = 8
    """minimal number of pixels for a light blob to be considered as a star"""

    star_point_min_bbox_ratio = 0.45
    """minimal ratio for the number of pixels belong to a star divided by the
    area of its bounding box"""

    star_point_max_num = 50
    """maximal number of star points"""

    star_point_disp = False
    """weather to visualize the star points for StarPointRefiner"""

    star_point_icp_max_dist_jump = 4.2
    """stop selecting points when distance increases by this factor compared to
    previous distance"""

    star_point_icp_max_iters = 100
    """max number of iterations for ICP in star point"""

    star_point_icp_stop_err = 0.15
    """average pixel error for the ICP to stop"""

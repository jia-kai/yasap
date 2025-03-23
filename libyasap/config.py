import argparse
import inspect
import ast
import itertools

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

                def chk_name(t):
                    if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store):
                        self.name2doc[t.id] = inspect.cleandoc(doc)

                if (isinstance(prev, ast.Assign) and
                        len(prev.targets) == 1):
                    chk_name(prev.targets[0])
                elif isinstance(prev, ast.AnnAssign):
                    chk_name(prev.target)
            prev = i


class ConfigWithArgparse:
    """a base class for configuration with argparse support"""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise AttributeError(f'invalid {k} for config {type(self)}')
            setattr(self, k, v)

    def update_from_args(self, args) -> "Self":
        for i in self._iter_config_keys():
            setattr(self, i, getattr(args, i))
        return self

    @classmethod
    def _iter_config_keys(cls):
        for k, v in cls.__dict__.items():
            if not k.startswith('_') and not callable(v):
                yield k

    @classmethod
    def add_to_parser(cls, parser: argparse.ArgumentParser):
        """add class attributes to parser"""
        lines = inspect.getsourcelines(cls)[0]
        num_space = min(
            sum(1 for _ in itertools.takewhile(str.isspace, l))
            for l in lines if (l and not l.isspace())
        )
        lines = [i[num_space:] for i in lines]
        src = '\n'.join(lines)
        doc = DocStringExtractor()
        doc.visit(ast.parse(src))
        for i in cls._iter_config_keys():
            kw = dict(help=f'{cls.__qualname__}: {doc.name2doc[i]}')
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

    use_xfeat = False
    """whether to use xfeat for coarse matching"""

    apply_lens_correction = False
    """whether to apply lens correction on the input images"""

    lens_correction_dist = 1e5
    """assumed object distance for lens correction"""

    ftr_match_coord_dist = 0.01
    """maximal distance of coordinates between matched feature points, relative
    to image width"""

    mask_moving = False
    """whether the mask should be moved to a coarse position on the current
    image by using previous transform"""

    draw_matches = False
    """whether to draw coarse match points"""

    draw_matches_no_wait = False
    """whether to not wait key when drawing match points"""

    linear_trans_before_ftr_match = False
    """whether to apply a linear transform to the gray image before feature
    computation and matching"""

    homography_rotation_interpolation = False
    """whether to enable interpolation of the rotation in homography; requires
    --use-restricted-transform"""

    save_refined_dir = ''
    """directory to save refined images"""

    skip_coarse_align = False
    """whether to skip corse alignment process; useful for aligning the
    foreground using optical flow (which is at roughly the same location on
    every image)"""

    align_err_disp_thresh = float('inf')
    """when refined alignment error is above this value, display the match
    points"""

    hessian_thresh = 800
    """Hessian threshold for SURF"""

    crop_all_imgs: str = ''
    """set as x:y:w:h to crop all input images"""

    remove_bg = False
    """remove background in alignment before further processing"""

    remove_bg_thresh = 0.7
    """threshold of quantile for background removal"""

    preproc_brightness = 0.0
    """brightness change during preprocessing"""

    preproc_contrast = 1.0
    """contrast change during preprocessing"""

    preproc_gaussian: float = 0.0
    """gaussian kernel std for blurring"""

    preproc_show = False
    """whether to show preprocessed image"""

    star_quality_filter: float = 0.0
    """if positive, will use star-point quality to filter images"""

    sparse_opt_quality_level = 0.3
    """quality level for feature selection in sparse optical flow"""

    sparse_opt_block_size = 7
    """block size for feature selection in sparse optical flow"""

    sparse_opt_levels = 1
    """pyramid levels of tracking in sparse optical flow"""

    sparse_opt_win_size = 12
    """window size for optical flow in sparse optical flow"""

    dense_opt_point_thresh = 0.99
    """select the point whose brightness is above this quantile"""

    max_ftr_points = 2000
    """maximum number of feature points to be used for coarse matching"""

    use_identity_trans = False
    """use identity transform to denoise foreground image"""

    refine_iters: int = 1
    """number of refinement iterations"""

    refine_abort_thresh = .5
    """threshold for giving up on the current image"""

    star_point_quantile = 0.99
    """quantile for deciding the threshold to find star points"""

    star_point_min_area = 8
    """minimal number of pixels for a light blob to be considered as a star"""

    star_point_min_bbox_ratio = 0.45
    """minimal ratio for the number of pixels belong to a star divided by the
    area of its bounding box"""

    star_point_max_num = 80
    """maximal number of star points"""

    star_point_disp = False
    """whether to visualize the star points for StarPointRefiner"""

    star_point_disp_no_wait = False
    """whether not to wait for keyboard when displaying star points"""

    star_point_icp_max_dist_jump = 4.2
    """stop selecting points when distance increases by this factor compared to
    previous distance"""

    star_point_icp_max_iters = 100
    """max number of iterations for ICP in star point"""

    star_point_icp_stop_err = 0.1
    """average pixel error for the ICP to stop"""

    star_point_refresh_ref_img: int = -1
    """set to a positive number to refresh the reference image after this number
    of iterations; useful for untracked imaging"""

    star_point_refresh_ref_err_thresh: float = 0.45
    """error threshold for setting an image as a reference image"""

    star_point_quality_max_drop = float('inf')
    """abandon the current image if the quality of star points compared to the
    first image is below this threshold"""

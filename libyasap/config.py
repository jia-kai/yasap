class AlignmentConfig:
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

    align_err_disp_threh = float('inf')
    """when refined alignment error is above this value, display the match
    points"""

    hessian_thresh = 800
    """Hessian threshold for SURF"""

    max_ftr_points = 2000
    """maximum number of feature points to be used for coarse matching"""

    use_identity_trans = False
    """use identity transform to denoise foreground image"""

    refine_abort_thresh = 1.8
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

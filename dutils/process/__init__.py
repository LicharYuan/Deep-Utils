from .bbox import points_in_bboxes, cal_ious_2d
from .image import undistort, resize_img, normalize


class TusimpleEvalConti(object):
    def __init__(self, dataloader, outdir, H_in, W_in, H_out, W_out, logger, half=False):
        self.dataloader = dataloader
        self.outdir = outdir
        # self.visdir = os.path.join(outdir, "vis")
        # check_path(self.visdir)

        self.half = half
        self.tensor_type = torch.float16 if half else torch.float32
        self.logger = logger
        self.H_in = H_in
        self.W_in = W_in
        self.H_out = H_out
        self.W_out = W_out
        self._preprocess(H_in, W_in, H_out, W_out)

    def _preprocess(self, H_in, W_in, H_out, W_out, dtype=torch.float32):
        K_s, K_d, T_gs_s, T_gs_d, \
        grid_template_in, grid_template_out = prepare_warping_transforms(H_in, W_in, H_out, W_out, torch.float32)
        self.K_s = K_s.cuda()
        self.K_d = K_d.cuda()
        self.K_s_inv = torch.inverse(K_s).cuda()
        self.K_d_inv = torch.inverse(K_d).cuda()
        self.T_gs_s = T_gs_s.cuda()
        self.T_gs_d = T_gs_d.cuda()
        self.grid_template_in = grid_template_in.cuda()
        self.grad_template_out = grid_template_out.cuda()

'''
Notes:
1. All of your implementation should be in this file. This is the ONLY .py file you need to edit & submit. 
2. Please Read the instructions and do not modify the input and output formats of function stitch_background() and panorama().
3. If you want to show an image for debugging, please use show_image() function in util.py. 
4. Please do NOT save any intermediate files in your final submission.
'''
import torch
import kornia as K
from typing import Dict
from utils import show_image

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''

# ------------------------------------ Task 1 ------------------------------------ #
def stitch_background(imgs: Dict[str, torch.Tensor]):
    """
    Args:
        imgs: input images are a dict of 2 images of torch.Tensor represent an input images for task-1.
    Returns:
        img: stitched_image: torch.Tensor of the output image.
    """
    img = torch.zeros((3, 256, 256)) # assumed 256*256 resolution. Update this as per your logic.

    #TODO: Add your code here. Do not modify the return and input arguments.
    keys = list(imgs.keys())
    img1 = imgs[keys[0]].float() / 255.0
    img2 = imgs[keys[1]].float() / 255.0

    img1b = img1.unsqueeze(0)
    img2b = img2.unsqueeze(0)

    gray1 = K.color.rgb_to_grayscale(img1b)
    gray2 = K.color.rgb_to_grayscale(img2b)

    sift = K.feature.SIFTFeature(800)
    laf1, resp1, desc1 = sift(gray1)
    laf2, resp2, desc2 = sift(gray2)

    matcher = K.feature.DescriptorMatcher("snn", 0.8)
    dists, idxs = matcher(desc1[0], desc2[0])

    pts1 = K.feature.get_laf_center(laf1)[0][idxs[:, 0]]
    pts2 = K.feature.get_laf_center(laf2)[0][idxs[:, 1]]

    weights = torch.ones(1, pts1.shape[0], dtype=pts1.dtype, device=pts1.device)
    H = K.geometry.homography.find_homography_dlt_iterated(
        pts2.unsqueeze(0), pts1.unsqueeze(0), weights
    )

    h1, w1 = img1.shape[1:]
    h2, w2 = img2.shape[1:]

    corners1 = torch.tensor(
        [[0.0, 0.0], [w1 - 1.0, 0.0], [w1 - 1.0, h1 - 1.0], [0.0, h1 - 1.0]],
        dtype=img1.dtype
    )
    corners2 = torch.tensor(
        [[0.0, 0.0], [w2 - 1.0, 0.0], [w2 - 1.0, h2 - 1.0], [0.0, h2 - 1.0]],
        dtype=img2.dtype
    )

    corners2h = K.geometry.conversions.convert_points_to_homogeneous(corners2).t()
    warped2h = H[0] @ corners2h
    warped2 = (warped2h[:2] / warped2h[2:]).t()

    all_corners = torch.cat([corners1, warped2], dim=0)

    min_xy = torch.floor(all_corners.min(dim=0).values)
    max_xy = torch.ceil(all_corners.max(dim=0).values)

    tx = -min_xy[0]
    ty = -min_xy[1]

    out_w = int((max_xy[0] - min_xy[0]).item() + 1)
    out_h = int((max_xy[1] - min_xy[1]).item() + 1)

    T = torch.tensor(
        [[1.0, 0.0, tx.item()], [0.0, 1.0, ty.item()], [0.0, 0.0, 1.0]],
        dtype=img1.dtype
    ).unsqueeze(0)

    eye = torch.eye(3, dtype=img1.dtype).unsqueeze(0)

    warp1 = K.geometry.transform.warp_perspective(img1b, T @ eye, (out_h, out_w))
    warp2 = K.geometry.transform.warp_perspective(img2b, T @ H, (out_h, out_w))

    mask1 = K.geometry.transform.warp_perspective(
        torch.ones(1, 1, h1, w1, dtype=img1.dtype), T @ eye, (out_h, out_w)
    )
    mask2 = K.geometry.transform.warp_perspective(
        torch.ones(1, 1, h2, w2, dtype=img2.dtype), T @ H, (out_h, out_w)
    )

    overlap = (mask1 > 0.5) & (mask2 > 0.5)
    diff = (warp1 - warp2).abs().mean(dim=1, keepdim=True)
    same = overlap & (diff < 0.12)
    only1 = (mask1 > 0.5) & ~(mask2 > 0.5)
    only2 = (mask2 > 0.5) & ~(mask1 > 0.5)

    out = torch.zeros_like(warp1)
    out = torch.where(only1.expand_as(out), warp1, out)
    out = torch.where(only2.expand_as(out), warp2, out)
    avg = (warp1 + warp2) * 0.5
    out = torch.where(same.expand_as(out), avg, out)
    out = torch.where((overlap & ~same).expand_as(out), torch.maximum(warp1, warp2), out)

    img = (out[0].clamp(0, 1) * 255.0).to(torch.uint8) 
    return img

# ------------------------------------ Task 2 ------------------------------------ #
def panorama(imgs: Dict[str, torch.Tensor]):
    """
    Args:
        imgs: dict {filename: CxHxW tensor} for task-2.
    Returns:
        img: panorama,
        overlap: torch.Tensor of the output image.
    """
    img = torch.zeros((3, 256, 256), dtype=torch.uint8)
    overlap = torch.empty((3, 256, 256))
    keys = list(imgs.keys())
    images = [imgs[k].float() / 255.0 for k in keys]
    n = len(images)
    overlap = torch.eye(n, dtype=torch.int64)
    if n == 0:
        return img, overlap
    if n == 1:
        return (images[0].clamp(0, 1) * 255.0).to(torch.uint8), overlap
    resized = []
    for im in images:
        c, h, w = im.shape
        max_dim = max(h, w)
        if max_dim > 1200:
            scale = 1200.0 / float(max_dim)
            new_h = max(64, int(round(h * scale)))
            new_w = max(64, int(round(w * scale)))
            im = K.geometry.transform.resize(im.unsqueeze(0), (new_h, new_w)).squeeze(0)
        resized.append(im)
    images = resized
    device = images[0].device
    dtype = images[0].dtype
    sift = K.feature.SIFTFeature(700)
    matcher = K.feature.DescriptorMatcher("snn", 0.8)
    centers = []
    descs = []
    for im in images:
        gray = K.color.rgb_to_grayscale(im.unsqueeze(0))
        laf, resp, desc = sift(gray)
        centers.append(K.feature.get_laf_center(laf).squeeze(0))
        descs.append(desc.squeeze(0))
    def safe_project(H, pts):
        pts_h = K.geometry.conversions.convert_points_to_homogeneous(pts).t()  # 3xN
        warped = H @ pts_h
        denom = warped[2:3, :]
        eps = torch.tensor(1e-8, dtype=denom.dtype, device=denom.device)
        sign = torch.where(denom >= 0, torch.ones_like(denom), -torch.ones_like(denom))
        denom = torch.where(denom.abs() < eps, sign * eps, denom)
        return (warped[:2, :] / denom).t()
    def normalize_H(H):
        if H is None or (not torch.isfinite(H).all()):
            return None
        if H[2, 2].abs() > 1e-8:
            H = H / H[2, 2]
        return H
    def estimate_homography(src_pts, dst_pts):
        if src_pts.shape[0] < 4:
            return None
        w = torch.ones(1, src_pts.shape[0], dtype=src_pts.dtype, device=src_pts.device)
        H = K.geometry.homography.find_homography_dlt_iterated(
            src_pts.unsqueeze(0), dst_pts.unsqueeze(0), w
        )[0]
        H = normalize_H(H)
        return H
    def ransac_homography(src_pts, dst_pts, reproj_thresh=3.0, max_iter=250):
        m = src_pts.shape[0]
        if m < 4:
            return None, None, None
        best_H = None
        best_inliers = None
        best_count = 0
        best_err = 1e12
        for _ in range(max_iter):
            perm = torch.randperm(m, device=src_pts.device)
            idx = perm[:4]
            H_try = estimate_homography(src_pts[idx], dst_pts[idx])
            if H_try is None:
                continue
            proj = safe_project(H_try, src_pts)
            if not torch.isfinite(proj).all():
                continue
            err = torch.norm(proj - dst_pts, dim=1)
            inliers = err < reproj_thresh
            count = int(inliers.sum().item())
            if count < 4:
                continue
            mean_err = err[inliers].mean().item()
            if (count > best_count) or (count == best_count and mean_err < best_err):
                best_count = count
                best_err = mean_err
                best_H = H_try
                best_inliers = inliers
        if best_H is None or best_inliers is None or best_count < 4:
            return None, None, None
        H_refined = estimate_homography(src_pts[best_inliers], dst_pts[best_inliers])
        if H_refined is None:
            return None, None, None
        proj = safe_project(H_refined, src_pts)
        err = torch.norm(proj - dst_pts, dim=1)
        inliers = err < reproj_thresh
        if int(inliers.sum().item()) >= 4:
            H_refined2 = estimate_homography(src_pts[inliers], dst_pts[inliers])
            if H_refined2 is not None:
                H_refined = H_refined2
                proj = safe_project(H_refined, src_pts)
                err = torch.norm(proj - dst_pts, dim=1)
                inliers = err < reproj_thresh
        if int(inliers.sum().item()) < 4:
            return None, None, None
        mean_err = err[inliers].mean().item()
        return H_refined, inliers, mean_err
    def homography_is_plausible(H):
        if H is None or (not torch.isfinite(H).all()):
            return False
        if torch.det(H).abs() < 1e-8:
            return False
        if torch.norm(H) > 1e4:
            return False
        if H[2, :2].abs().max().item() > 0.02:
            return False
        return True
    pair_H = {}
    pair_score = torch.zeros((n, n), dtype=torch.float32)
    for i in range(n):
        for j in range(i + 1, n):
            if descs[i].shape[0] < 12 or descs[j].shape[0] < 12:
                continue
            dists, idxs = matcher(descs[i], descs[j])
            if idxs.shape[0] < 12:
                continue
            pts_i = centers[i][idxs[:, 0]]
            pts_j = centers[j][idxs[:, 1]]
            H_ij, inliers, mean_err = ransac_homography(
                pts_i, pts_j, reproj_thresh=3.0, max_iter=250
            )
            if H_ij is None or inliers is None:
                continue
            if not homography_is_plausible(H_ij):
                continue
            inlier_count = int(inliers.sum().item())
            inlier_ratio = float(inliers.float().mean().item())
            if inlier_count < 18:
                continue
            if inlier_ratio < 0.22:
                continue
            if mean_err > 2.8:
                continue
            H_ji = normalize_H(torch.linalg.inv(H_ij))
            if not homography_is_plausible(H_ji):
                continue
            overlap[i, j] = 1
            overlap[j, i] = 1
            pair_H[(i, j)] = H_ij
            pair_H[(j, i)] = H_ji
            score = float(inlier_count) / (mean_err + 1e-6)
            pair_score[i, j] = score
            pair_score[j, i] = score
    visited = [False] * n
    components = []
    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in range(n):
                if overlap[u, v].item() == 1 and not visited[v]:
                    visited[v] = True
                    stack.append(v)
        components.append(comp)
    usable_components = [c for c in components if len(c) > 1]
    if len(usable_components) == 0:
        return (images[0].clamp(0, 1) * 255.0).to(torch.uint8), overlap
    main_comp = max(usable_components, key=len)
    main_comp = sorted(main_comp)
    best_ref = main_comp[0]
    best_ref_score = -1.0
    for u in main_comp:
        s = pair_score[u, main_comp].sum().item()
        if s > best_ref_score:
            best_ref_score = s
            best_ref = u
    ref = best_ref
    tree_parent = {ref: None}
    in_tree = {ref}
    while len(in_tree) < len(main_comp):
        best_u = None
        best_v = None
        best_w = -1.0
        for u in in_tree:
            for v in main_comp:
                if v in in_tree:
                    continue
                w = pair_score[u, v].item()
                if w > best_w and (v, u) in pair_H:
                    best_w = w
                    best_u = u
                    best_v = v
        if best_v is None:
            break
        tree_parent[best_v] = best_u
        in_tree.add(best_v)
    transforms = {ref: torch.eye(3, dtype=dtype, device=device)}
    changed = True
    while changed:
        changed = False
        for v in list(tree_parent.keys()):
            if v in transforms:
                continue
            u = tree_parent[v]
            if u is None or u not in transforms:
                continue
            if (v, u) not in pair_H:
                continue
            H_v_to_ref = transforms[u] @ pair_H[(v, u)]
            H_v_to_ref = normalize_H(H_v_to_ref)
            if H_v_to_ref is not None and homography_is_plausible(H_v_to_ref):
                transforms[v] = H_v_to_ref
                changed = True
    usable = sorted(transforms.keys())
    if len(usable) == 0:
        return (images[ref].clamp(0, 1) * 255.0).to(torch.uint8), overlap
    if len(usable) == 1:
        return (images[usable[0]].clamp(0, 1) * 255.0).to(torch.uint8), overlap
    all_corners = []
    kept = []
    for idx in usable:
        im = images[idx]
        h, w = im.shape[1:]
        corners = torch.tensor(
            [[0.0, 0.0], [w - 1.0, 0.0], [w - 1.0, h - 1.0], [0.0, h - 1.0]],
            dtype=dtype,
            device=device
        )
        warped_corners = safe_project(transforms[idx], corners)
        if not torch.isfinite(warped_corners).all():
            continue
        width_est = (warped_corners[1] - warped_corners[0]).norm().item()
        height_est = (warped_corners[3] - warped_corners[0]).norm().item()
        if width_est < 20 or height_est < 20:
            continue
        if width_est > 6 * w or height_est > 6 * h:
            continue
        all_corners.append(warped_corners)
        kept.append(idx)
    usable = kept
    if len(all_corners) == 0:
        return (images[ref].clamp(0, 1) * 255.0).to(torch.uint8), overlap
    all_corners = torch.cat(all_corners, dim=0)
    min_xy = torch.floor(all_corners.min(dim=0).values)
    max_xy = torch.ceil(all_corners.max(dim=0).values)
    out_w = int((max_xy[0] - min_xy[0]).item() + 1)
    out_h = int((max_xy[1] - min_xy[1]).item() + 1)
    if out_w <= 0 or out_h <= 0:
        return (images[ref].clamp(0, 1) * 255.0).to(torch.uint8), overlap
    max_side = 2200
    if out_w > max_side or out_h > max_side:
        scale = min(float(max_side) / float(out_w), float(max_side) / float(out_h))
        out_w = max(64, int(round(out_w * scale)))
        out_h = max(64, int(round(out_h * scale)))
    else:
        scale = 1.0
    tx = -min_xy[0].item()
    ty = -min_xy[1].item()
    T = torch.tensor(
        [[scale, 0.0, scale * tx],
         [0.0, scale, scale * ty],
         [0.0, 0.0, 1.0]],
        dtype=dtype,
        device=device
    )
    acc = torch.zeros((1, 3, out_h, out_w), dtype=dtype, device=device)
    cnt = torch.zeros((1, 1, out_h, out_w), dtype=dtype, device=device)
    for idx in usable:
        im = images[idx].unsqueeze(0)
        h, w = images[idx].shape[1:]
        H = normalize_H(T @ transforms[idx]).unsqueeze(0)
        warped = K.geometry.transform.warp_perspective(im, H, (out_h, out_w))
        mask_in = torch.ones((1, 1, h, w), dtype=dtype, device=device)
        mask = K.geometry.transform.warp_perspective(mask_in, H, (out_h, out_w))
        valid = (mask > 0.5).float()
        acc = acc + warped * valid
        cnt = cnt + valid
    if (cnt > 0).sum().item() == 0:
        return (images[ref].clamp(0, 1) * 255.0).to(torch.uint8), overlap
    out = acc / cnt.clamp_min(1e-6)
    img = (out[0].clamp(0, 1) * 255.0).to(torch.uint8)

    return img, overlap
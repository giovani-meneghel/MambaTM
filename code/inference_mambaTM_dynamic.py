import argparse
from PIL import Image
import sys
import os
import time
import subprocess
start_time = time.time()
print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import cv2
import numpy as np
import json
import torch
from TM_model import Model
import torchvision.transforms.functional as TF

def get_video_fps_rational(video_path):
    cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate,avg_frame_rate',
        '-of', 'json', video_path
    ]
    try:
        output = subprocess.check_output(cmd, text=True).strip()
        data = json.loads(output)
        streams = data.get('streams', [])
        if streams:
            for key in ('r_frame_rate', 'avg_frame_rate'):
                rate = streams[0].get(key, '')
                if rate and rate != '0/0':
                    if '/' in rate:
                        num, den = map(int, rate.split('/'))
                        if den != 0:
                            val = num / den
                            if abs(val - round(val)) < 0.005:
                                return f"{int(round(val))}/1"
                            for n, d in [(24000, 1001), (30000, 1001), (48000, 1001), (60000, 1001), (120000, 1001), (240000, 1001)]:
                                if abs(val - n / d) < 0.005:
                                    return f"{n}/{d}"
                            return f"{num}/{den}"
    except Exception as e:
        print(f"Warning: Could not get FPS via ffprobe: {e}")

    try:
        cap = cv2.VideoCapture(video_path)
        fps_cv = float(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        if fps_cv > 0:
            if abs(fps_cv - round(fps_cv)) < 0.005:
                return f"{int(round(fps_cv))}/1"
            for n, d in [(24000, 1001), (30000, 1001), (48000, 1001), (60000, 1001), (120000, 1001), (240000, 1001)]:
                if abs(fps_cv - n / d) < 0.005:
                    return f"{n}/{d}"
            return str(fps_cv)
    except Exception as e:
        print(f"Warning: Could not get FPS via cv2: {e}")

    raise RuntimeError(f"Could not determine valid FPS for '{video_path}'")

def get_args():
    parser = argparse.ArgumentParser(description='test the DATUM on images and restoration') 
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a specific .pth file path')
    parser.add_argument('--input_path', type=str, default=None, help='path of input video')
    parser.add_argument('--out_path', type=str, default=None, help='path of output video')
    parser.add_argument('--start_frame', type=int, default=0, help='start index of frames')
    parser.add_argument('--num_frames', type=int, default=0, help='max number of frames (0 or negative to process all frames)')
    parser.add_argument('--clip', type=int, default=120, help='max number of frames in a temporal patch')
    parser.add_argument('--ps', type=int, default=256, help='max spatial dimensions for a patch')
    parser.add_argument('--model', type=str, default='MambaTM', help='type of model to construct')
    parser.add_argument('--version', type=str, default="v2")
    parser.add_argument('--n_features', type=int, default=16, help='base # of channels for Conv')
    parser.add_argument('--n_blocks', type=int, default=6, help='# of blocks in middle part of the model')
    parser.add_argument('--future_frames', type=int, default=2, help='use # of future frames')
    parser.add_argument('--past_frames', type=int, default=2, help='use # of past frames')
    parser.add_argument('--activation', type=str, default='gelu', help='activation function')
    parser.add_argument('--resize', type=float, default=1.0, help='resize ratio')
    
    # New exposed user-friendly parameters
    parser.add_argument('--dynamic', action='store_true',
                        help='Use dynamic model weights (MambaTM_dynamic.pth). Best for scenes with moving objects/people. Softer, more conservative geometric stabilization to prevent motion artifacts.')
    parser.add_argument('--static', action='store_true',
                        help='Use static model weights (MambaTM_static.pth). Best for stationary camera scenes. Stronger, more aggressive stabilization and de-warping.')
    parser.add_argument('--patch-size', type=str, default=None,
                        help="Spatial patch size. Can be a numeric value (e.g. 256) or one of 'low' (128), 'mid' (256), or 'high' (512). Larger values give stronger, more globally coherent de-warping but consume more VRAM.")
    parser.add_argument('--clip-size', type=int, default=None,
                        help="Temporal clip size (suggest default: 120, auto-capped if VRAM is limited). Sets the temporal chunk length for the model. Larger values give stronger temporal stabilization but use more VRAM.")
    parser.add_argument('--reverse', action='store_true',
                        help='Process the video in reverse order temporally, then reverse the output back to normal. This moves the initial recurrent warm-up phase to the end of the video, resulting in better stabilization at the beginning.')
    return parser.parse_args()

def split_to_patches(h, w, s):
    nh = h // s + 1
    nw = w // s + 1
    if nh > 1:
        ol_h = int((nh * s - h) / (nh - 1))
        h_start = 0
        hpos = [h_start]
        for i in range(1, nh):
            h_start = hpos[-1] + s - ol_h
            if h_start+s > h:
                h_start = h-s
            hpos.append(h_start)      
        if len(hpos)==2 and hpos[0] == hpos[1]:
            hpos = [hpos[0]]
    else:
        hpos = [0]
    if nw > 1:
        ol_w = int((nw * s - w) / (nw - 1))
        w_start = 0  
        wpos = [w_start]
        for i in range(1, nw):
            w_start = wpos[-1] + s - ol_w
            if w_start+s > w:
                w_start = w-s
            wpos.append(w_start)
        if len(wpos)==2 and wpos[0] == wpos[1]:
            wpos = [wpos[0]]
    else:
        wpos = [0]
    return hpos, wpos
    
def test_spatial_overlap(input_blk, model, patch_size, chunk_idx=0, n_chunks=1):
    _,l,c,h,w = input_blk.shape
    hpos, wpos = split_to_patches(h, w, patch_size)
    out_spaces = torch.zeros(_,l,c,h,w).cuda()
    out_masks = torch.zeros(_,l,c,h,w).cuda()
    n_patches = len(hpos) * len(wpos)
    patch_idx = 0
    for hi in hpos:
        for wi in wpos:
            patch_idx += 1
            if h > patch_size:
                h_end = hi+patch_size
            else:
                h_end = h
            if w > patch_size:
                w_end = wi+patch_size
            else:
                w_end = w
            patch_t0 = time.time()
            print(f"  [Chunk {chunk_idx}/{n_chunks}] Spatial patch {patch_idx}/{n_patches}  "
                  f"(y:{hi}-{h_end}, x:{wi}-{w_end})  patch_size={patch_size}x{patch_size}...",
                  flush=True)
            input_ = input_blk[..., hi:h_end, wi:w_end]
            output_ = model(input_)
            if isinstance(output_, tuple):
                output_ = output_[0]
            out_spaces[..., hi:h_end, wi:w_end].add_(output_)
            out_masks[..., hi:h_end, wi:w_end].add_(torch.ones_like(output_))
            print(f"    -> patch done in {time.time()-patch_t0:.1f}s", flush=True)
    return out_spaces / out_masks
    
def temp_segment(total_frames, chunk_len, valid_len):
    residual = (chunk_len - valid_len) // 2
    test_frame_info = [{'start':0, 'range':[0,chunk_len-residual]}]
    num_chunk = (total_frames-1) // valid_len
    for i in range(1, num_chunk+1):
        if i == num_chunk:
            test_frame_info.append({'start':total_frames-chunk_len, 'range':[i*valid_len+residual,total_frames]})
        elif i*valid_len+chunk_len >= total_frames:
            test_frame_info.append({'start':total_frames-chunk_len, 'range':[i*valid_len+residual, total_frames]})
            break
        else:
            test_frame_info.append({'start':i*valid_len, 'range':[i*valid_len+residual,i*valid_len+chunk_len-residual]})
    return len(test_frame_info), test_frame_info

def tensor2img(tensor, fidx):
    img = tensor[0, fidx, ...].data.squeeze().float().cpu().clamp_(0, 1).numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))  # CHW-RGB to HWC-BGR
    img = (img * 255.0).round().astype(np.uint8)  # float32 to uint8
    return img   
    
    
args = get_args()

# Handle --dynamic and --static model weight overrides
script_dir = os.path.dirname(os.path.abspath(__file__))
if args.dynamic and args.static:
    raise ValueError("Cannot specify both --dynamic and --static. Please choose one.")
elif args.dynamic:
    args.load = os.path.join(script_dir, "model_zoo", "MambaTM_dynamic.pth")
    print(f"Option --dynamic specified. Overriding model path: {args.load}")
elif args.static:
    args.load = os.path.join(script_dir, "model_zoo", "MambaTM_static.pth")
    print(f"Option --static specified. Overriding model path: {args.load}")
elif not args.load:
    # If neither is specified and load is not provided, default to dynamic weights
    args.load = os.path.join(script_dir, "model_zoo", "MambaTM_dynamic.pth")
    print(f"No weight override specified. Defaulting to dynamic weights: {args.load}")

# Handle --patch-size argument override
if args.patch_size is not None:
    ps_lower = args.patch_size.lower()
    if ps_lower == 'low':
        args.ps = 128
    elif ps_lower == 'mid':
        args.ps = 256
    elif ps_lower == 'high':
        args.ps = 512
    else:
        try:
            args.ps = int(args.patch_size)
        except ValueError:
            raise ValueError(f"Invalid value for --patch-size: '{args.patch_size}'. Must be 'low', 'mid', 'high' or an integer.")
    print(f"Option --patch-size specified. Setting patch size (ps) to: {args.ps}")

# Handle --clip-size argument override
if args.clip_size is not None:
    args.clip = args.clip_size
    print(f"Option --clip-size specified. Setting temporal clip size (clip) to: {args.clip}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Model(args).cuda()
checkpoint = torch.load(args.load)
model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint.keys() else checkpoint)

max_frames = args.num_frames
start_frame = args.start_frame
resize = args.resize
# 'G00393_set1_rand_1622563286641_bd4061b9'
input_path = args.input_path
output_path = args.out_path
# clip_path = './results/video_19_in.mp4'
with torch.no_grad():
    # Detect GPU capability for autocast dtype (BF16 for RTX 4090, FP16 for RTX 2060)
    dtype = torch.float16
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        dtype = torch.bfloat16
        print("Device supports BF16 (RTX 4090/Ada Lovelace detected). Using BFloat16 mixed precision.")
    else:
        print("Device does not support BF16 natively (RTX 2060 or older detected). Using Float16 mixed precision.")

    with torch.amp.autocast('cuda', enabled=True, dtype=dtype):
        turb_vid = cv2.VideoCapture(input_path)
        h, w = int(turb_vid.get(4)), int(turb_vid.get(3))
        fps = get_video_fps_rational(input_path)
        print(f"Video resolution: {w}x{h}, FPS: {fps}")
        total_frames = int(turb_vid.get(cv2.CAP_PROP_FRAME_COUNT))
        turb_vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        all_frames = []
        if max_frames is not None and max_frames > 0:
            frames_to_read = max_frames
            if total_frames > 0:
                frames_to_read = min(frames_to_read, max(0, total_frames - start_frame))
            for _ in range(frames_to_read):
                ret, frame = turb_vid.read()
                if not ret or frame is None:
                    break
                all_frames.append(frame)
        else:
            while True:
                ret, frame = turb_vid.read()
                if not ret or frame is None:
                    break
                all_frames.append(frame)
        turb_vid.release()
        
        if resize<1:
            w = int(w*resize)
            h = int(h*resize)
            all_frames = [cv2.resize(f, (w, h)) for f in all_frames]
        total_frames = len(all_frames)
        print(f"video {input_path}, input frames {total_frames}, h {h} x w {w}")
        
        if args.reverse:
            print("Option --reverse specified. Reversing the input frames temporally for backward stabilization.")
            all_frames = all_frames[::-1]

        # ---- Auto-detect optimal --clip based on GPU VRAM and patch size ----
        # The Mamba blocks process sequences of length L = clip x (ps/8)^2 after
        # the encoder 8x downscale.  The Kogge-Stone parallel scan (fast path)
        # has O(L*log L) memory; sequences above _KS_L_THRESHOLD (40,000) fall
        # into the slow JIT sequential path.  We auto-cap clip so L stays in
        # the fast path and fits comfortably in VRAM.
        cl = args.clip
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            spatial_per_frame = (args.ps // 8) ** 2  # 32x32 = 1024 for ps=256
            # Cap 1: keep sequence length under the fast Kogge-Stone threshold
            KS_THRESHOLD = 40_000
            ks_max_clip = max(10, KS_THRESHOLD // spatial_per_frame - 1)
            # Cap 2: keep Kogge-Stone VRAM under the available budget
            # Benchmarked: ~0.1057 MB per sequence element at (B=1, D=256, N=16)
            MB_PER_ELEMENT = 1298.0 / 12288.0
            available_mb = max(0, (vram_gb - 1.5)) * 1024  # reserve 1.5 GB for model+activations
            vram_max_clip = max(10, int(available_mb / MB_PER_ELEMENT / spatial_per_frame))
            recommended_clip = min(ks_max_clip, vram_max_clip, cl)
            if recommended_clip < cl:
                print(f"Auto-adjusting --clip from {cl} to {recommended_clip} for GPU "
                      f"({torch.cuda.get_device_name(0)}, {vram_gb:.1f} GB VRAM).  "
                      f"Seq length per patch: {recommended_clip * spatial_per_frame} "
                      f"(threshold: {KS_THRESHOLD})")
                cl = recommended_clip

        if total_frames > cl:
            n_chunks, test_frame_info = temp_segment(total_frames, chunk_len=cl, valid_len=cl)
        else:
            n_chunks, test_frame_info = temp_segment(total_frames, chunk_len=total_frames, valid_len=total_frames)
            cl = total_frames
            
        out_frames = []
        frame_idx = 0
        patch_unit = 8
        if h%patch_unit==0:
            nh = h
        else:
            nh = h//patch_unit*patch_unit + patch_unit
        if w%patch_unit==0:
            nw = w
        else:
            nw = w//patch_unit*patch_unit + patch_unit 
        padw, padh = nw-w, nh-h
     
        for i in range(n_chunks):
            out_range = test_frame_info[i]['range']
            in_range = [test_frame_info[i]['start'], test_frame_info[i]['start']+cl]
            chunk_t0 = time.time()
            print(f"\n[Chunk {i+1}/{n_chunks}] Processing frames {in_range[0]}-{min(in_range[1], total_frames)-1}  "
                  f"(output range: {out_range[0]}-{out_range[1]-1})...", flush=True)
            inp_imgs = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in all_frames[in_range[0]:in_range[1]]]
            inp_imgs = [TF.pad(img, (0,0,padw,padh), padding_mode='reflect') for img in inp_imgs]
            inp_imgs = [TF.to_tensor(img) for img in inp_imgs]
            input_ = torch.stack(inp_imgs, dim=0).unsqueeze(0).cuda()
            if max(h,w)>args.ps:
                output = test_spatial_overlap(input_, model, args.ps, chunk_idx=i+1, n_chunks=n_chunks)
            else:
                print(f"  Running full-frame inference (no spatial tiling needed)...", flush=True)
                output,_ = model(input_)
            # Determine which frames of the model output to keep.
            # out_range[0] is the global index of the first frame we want;
            # in_range[0] is where the input chunk started -> the offset within
            # the model output tensor is simply (out_range[0] - in_range[0]).
            # The previous code had a hard-coded "-2" that caused range() to go
            # negative and write 2 extra (repeated) frames per chunk.
            j_start = out_range[0] - in_range[0]   # correct offset, always >= 0
            assert 0 <= j_start < output.shape[1], \
                f"Bad j_start={j_start} for out_range={out_range}, in_range={in_range}"
            for j in range(j_start, output.shape[1]):
                out = cv2.cvtColor(tensor2img(output, j), cv2.COLOR_RGB2BGR)
                out_frames.append(out)
            print(f"[Chunk {i+1}/{n_chunks}] Done in {time.time()-chunk_t0:.1f}s  "
                  f"(wrote frames {out_range[0]}-{out_range[0]+output.shape[1]-j_start-1}, "
                  f"total output frames so far: {len(out_frames)})", flush=True)
            torch.cuda.empty_cache()
                
        print(f"video {input_path} done! input frames {total_frames}, output frames {len(out_frames)}")
        if args.reverse:
            print("Reversing the output frames back to normal temporal order.")
            out_frames = out_frames[::-1]
        print(f"Encoding output video with FFmpeg (libx264, CRF 17, preset slow, yuv420p, +faststart)...")
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-v', 'error',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{w}x{h}', '-pix_fmt', 'bgr24', '-r', str(fps),
            '-i', '-',
            '-c:v', 'libx264', '-crf', '17', '-preset', 'slow', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            '-an', output_path
        ]
        proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
        for fid, frame in enumerate(out_frames):
            proc.stdin.write(frame[:h, :w, :].tobytes())
        proc.stdin.close()
        proc.wait()
        if proc.returncode != 0:
            print(f"Error: FFmpeg encoding failed with return code {proc.returncode}")

end_time = time.time()
total_time = end_time - start_time
print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
print(f"Total running time: {total_time:.2f} seconds")

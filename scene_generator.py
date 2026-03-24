"""Enhanced synthetic road scene generator for DL perception testing."""
import numpy as np
import cv2


def generate_road_scene(width=1280, height=720, time_of_day="day",
                        weather="clear", seed=None):
    if seed is not None:
        np.random.seed(seed)

    scene = np.zeros((height, width, 3), dtype=np.uint8)
    hy    = int(height * 0.42)

    SKY = {"day":  ((100,160,220),(170,210,240)),
           "dusk": ((20,10,60),(180,80,40)),
           "night":((5,5,15),(10,10,30))}
    sky_top, sky_bot = [np.array(c) for c in SKY.get(time_of_day, SKY["day"])]
    for y in range(hy):
        t = y/hy
        scene[y,:] = ((1-t)*sky_top + t*sky_bot).astype(np.uint8)

    if time_of_day=="night":
        sy,sx = np.random.randint(0,hy,200), np.random.randint(0,width,200)
        for sy_,sx_ in zip(sy,sx):
            b = np.random.randint(150,255)
            scene[sy_,sx_] = [b,b,b]

    vp_x = width//2 + np.random.randint(-60,60)
    road_col = np.array([60,60,65]) if time_of_day!="night" else np.array([25,25,30])
    road_far  = np.array([80,80,85]) if time_of_day!="night" else np.array([15,15,20])
    grass_col = [60,110,50] if time_of_day!="night" else [20,40,20]

    for y in range(hy,height):
        t = (y-hy)/(height-hy)
        col = ((1-t)*road_far + t*road_col).astype(np.uint8)
        hw  = int(t*width*0.55)
        xl  = max(0,vp_x-hw); xr = min(width,vp_x+hw)
        scene[y,xl:xr] = col
        scene[y,:xl]   = grass_col
        scene[y,xr:]   = grass_col
        ew = max(1,int(t*5))
        scene[y,max(0,xl-ew):xl+2] = [200,200,200]
        scene[y,xr-2:min(width,xr+ew)] = [200,200,200]

    # Lane markings
    for offset in [-0.33,0,0.33]:
        for y in range(hy+10,height):
            t = (y-hy)/(height-hy)
            hw = int(t*width*0.55)
            lx = int(vp_x + offset*2*hw)
            ds = max(2,int(t*25)); dg = max(4,int(t*40))
            if (y//(ds+dg))%2==0:
                lw = max(1,int(t*5))
                col = [230,200,10] if offset==0 else [220,220,220]
                scene[y,lx-lw:lx+lw] = col

    # Trees
    for side in [-1,1]:
        for i in range(8):
            t = (i+1)/9.0
            yb = int(hy + t*(height-hy))
            hw = int(t*width*0.55)
            xb = vp_x + side*(hw+int(t*60))
            th = int(t*120); tw = int(t*35)
            trunk_w = max(2,int(t*8))
            cv2.rectangle(scene,(xb-trunk_w,yb-th),(xb+trunk_w,yb),(80,55,30),-1)
            canopy = (35,110,35) if time_of_day!="night" else (15,50,15)
            cv2.ellipse(scene,(xb,yb-th),(tw,int(th*0.6)),0,0,360,canopy,-1)

    # Signs
    for sp,col in [(0.25,(220,30,30)),(0.55,(255,255,255)),(0.80,(255,200,0))]:
        t = sp; yb = int(hy + t*(height-hy))
        hw = int(t*width*0.55); xb = vp_x+(hw+int(t*40))
        ph = int(t*90); sr = max(6,int(t*28))
        cv2.rectangle(scene,(xb-max(2,int(t*5)),yb-ph),(xb+max(2,int(t*5)),yb),(150,150,150),-1)
        cv2.circle(scene,(xb,yb-ph),sr,col,-1)
        cv2.circle(scene,(xb,yb-ph),sr,(255,255,255),2)

    # Lead vehicle
    t = 0.35; yv = int(hy + t*(height-hy))
    hw = int(t*width*0.55); cw = int(t*120); ch = int(t*60)
    cx_ = vp_x - hw//3
    body = (180,80,80) if time_of_day!="night" else (100,40,40)
    cv2.rectangle(scene,(cx_-cw//2,yv-ch),(cx_+cw//2,yv),body,-1)
    cv2.rectangle(scene,(cx_-cw//3,yv-ch-int(ch*0.5)),(cx_+cw//3,yv-ch+4),(150,60,60),-1)
    tlc = (80,30,200) if time_of_day=="night" else (255,80,80)
    cv2.rectangle(scene,(cx_-cw//2,yv-ch//3),(cx_-cw//2+int(cw*0.12),yv-int(ch*0.1)),tlc,-1)
    cv2.rectangle(scene,(cx_+cw//2-int(cw*0.12),yv-ch//3),(cx_+cw//2,yv-int(ch*0.1)),tlc,-1)

    # Weather
    if weather=="rain":
        for _ in range(800):
            rx,ry=np.random.randint(0,width),np.random.randint(0,height)
            rl=np.random.randint(8,20)
            cv2.line(scene,(rx,ry),(rx+2,ry+rl),(180,180,200),1)
        scene = cv2.addWeighted(scene,0.85,np.full_like(scene,[100,110,120]),0.15,0)
    elif weather=="fog":
        fog=np.full_like(scene,[200,210,220])
        alpha=np.linspace(0.1,0.7,height)
        for y_ in range(height):
            scene[y_] = cv2.addWeighted(
                scene[y_].reshape(1,-1,3),1-alpha[y_],
                fog[y_].reshape(1,-1,3),alpha[y_],0).reshape(-1,3)
    elif weather=="snow":
        for _ in range(400):
            sx_,sy_=np.random.randint(0,width),np.random.randint(0,height)
            cv2.circle(scene,(sx_,sy_),np.random.randint(1,4),(240,245,255),-1)

    noise = np.random.randint(-8,8,scene.shape,dtype=np.int16)
    scene = np.clip(scene.astype(np.int16)+noise,0,255).astype(np.uint8)
    return scene, {"vp_x":vp_x,"hy":hy}

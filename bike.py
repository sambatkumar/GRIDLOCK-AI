"""
GRID LOCK — Bike Game
=====================
Controls : W/S = accelerate/brake   A/D = steer left/right
Helmet ON : max speed 80 km/h
No Helmet : max speed capped at 30 km/h  +  warning overlay

Requirements:
    pip install pygame opencv-python ultralytics numpy

Model path: update MODEL_PATH below to point to your best.pt
"""

import sys, os, math, random, threading, time
import numpy as np

# ── Model path ────────────────────────────────────────────────────────────────
MODEL_PATH = "models/best.pt"

# ── Try imports ───────────────────────────────────────────────────────────────
try:
    import pygame
except ImportError:
    print("Run:  pip install pygame"); input(); sys.exit()
try:
    import cv2
except ImportError:
    print("Run:  pip install opencv-python"); input(); sys.exit()
try:
    from ultralytics import YOLO
except ImportError:
    print("Run:  pip install ultralytics"); input(); sys.exit()

# ── Constants ─────────────────────────────────────────────────────────────────
W, H          = 900, 650
FPS           = 60
CAM_W, CAM_H  = 200, 150       # webcam preview size

ROAD_W        = 300            # pixels wide
LANE_COLOR    = (40, 40, 40)
ROAD_COLOR    = (55, 55, 55)
STRIPE_COLOR  = (220, 200, 50)
GRASS_COLOR   = (30, 80, 20)
KERB_COLOR    = (200, 50, 50)

GREEN  = (50,  255, 80)
RED    = (255, 50,  50)
ORANGE = (255, 160, 20)
WHITE  = (240, 240, 240)
BLACK  = (10,  10,  10)
CYAN   = (0,   220, 220)
DARK   = (15,  15,  20)

SPEED_HELMET    = 80   # km/h max with helmet
SPEED_NO_HELMET = 30   # km/h max without helmet

# ── YOLO Detection Thread ─────────────────────────────────────────────────────
class HelmetDetector:
    def __init__(self):
        self.helmet_on    = True          # safe default
        self.confidence   = 0.0
        self.frame        = None          # latest webcam frame (BGR)
        self.label        = "Initializing..."
        self._running     = True
        self._lock        = threading.Lock()

        print("Loading YOLO model...")
        try:
            self.model = YOLO(MODEL_PATH)
            print("Model loaded ✅")
        except Exception as e:
            print(f"Model load failed: {e}")
            self.model = None

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Webcam not found — running in keyboard-only mode (H = toggle helmet)")
            with self._lock:
                self.label = "No webcam"
            return

        while self._running:
            ret, frame = cap.read()
            if not ret:
                continue

            display = cv2.resize(frame, (CAM_W, CAM_H))

            if self.model:
                results = self.model(frame, verbose=False)[0]
                best_conf   = 0.0
                best_helmet = True   # default to safe

                # Find highest-confidence detection
                for box in results.boxes:
                    conf  = float(box.conf[0])
                    cls   = int(box.cls[0])
                    name  = results.names[cls].lower()
                    if conf > best_conf:
                        best_conf = conf
                        best_helmet = "with" in name or "helmet" in name and "without" not in name

                with self._lock:
                    self.helmet_on  = best_helmet if best_conf > 0.4 else True
                    self.confidence = best_conf
                    self.label      = f"{'HELMET ON' if self.helmet_on else 'NO HELMET'}  {best_conf:.0%}"

                # Draw boxes on display frame
                for box in results.boxes:
                    conf = float(box.conf[0])
                    if conf < 0.4:
                        continue
                    cls  = int(box.cls[0])
                    name = results.names[cls].lower()
                    x1,y1,x2,y2 = box.xyxy[0].tolist()
                    # Scale to display size
                    sx = CAM_W / frame.shape[1]
                    sy = CAM_H / frame.shape[0]
                    color = (0,255,0) if ("with" in name and "without" not in name) else (0,0,255)
                    cv2.rectangle(display,
                                  (int(x1*sx), int(y1*sy)),
                                  (int(x2*sx), int(y2*sy)),
                                  color, 1)

            with self._lock:
                self.frame = display.copy()

            time.sleep(0.03)   # ~30 fps detection

        cap.release()

    def get_state(self):
        with self._lock:
            return self.helmet_on, self.confidence, self.label, self.frame

    def stop(self):
        self._running = False


# ── Road Stripe ───────────────────────────────────────────────────────────────
class Stripe:
    def __init__(self, y):
        self.y = y

    def update(self, speed):
        self.y += speed * 0.6
        if self.y > H + 40:
            self.y -= H + 120

    def draw(self, surf, road_x):
        cx = road_x + ROAD_W // 2
        pygame.draw.rect(surf, STRIPE_COLOR, (cx - 5, int(self.y), 10, 40))


# ── Obstacle ──────────────────────────────────────────────────────────────────
class Obstacle:
    TYPES = ["cone", "pothole", "barrel"]

    def __init__(self, road_x):
        self.x    = road_x + random.randint(30, ROAD_W - 30)
        self.y    = -40
        self.kind = random.choice(self.TYPES)
        self.w, self.h = 20, 20

    def update(self, speed):
        self.y += speed * 0.6

    def draw(self, surf):
        if self.kind == "cone":
            pts = [(self.x, self.y - 18),
                   (self.x - 12, self.y + 10),
                   (self.x + 12, self.y + 10)]
            pygame.draw.polygon(surf, ORANGE, pts)
            pygame.draw.rect(surf, WHITE, (self.x - 14, self.y + 8, 28, 6))
        elif self.kind == "pothole":
            pygame.draw.ellipse(surf, (25, 25, 25),
                                (self.x - 18, self.y - 10, 36, 20))
            pygame.draw.ellipse(surf, (15, 15, 15),
                                (self.x - 14, self.y - 7, 28, 14))
        else:  # barrel
            pygame.draw.rect(surf, (180, 60, 20),
                             (self.x - 10, self.y - 14, 20, 28), border_radius=4)
            pygame.draw.rect(surf, (220, 100, 40),
                             (self.x - 10, self.y - 5, 20, 6))

    def rect(self):
        return pygame.Rect(self.x - self.w//2, self.y - self.h//2,
                           self.w, self.h)


# ── Bike (player) ─────────────────────────────────────────────────────────────
class Bike:
    def __init__(self, road_x):
        self.x     = road_x + ROAD_W // 2
        self.y     = H - 140
        self.speed = 0.0      # pixels/frame scroll speed
        self.kmh   = 0.0
        self.angle = 0.0      # visual lean
        self.score = 0
        self.alive = True
        self.hit_timer = 0

    def update(self, keys, max_kmh, dt):
        accel = 0.15
        brake = 0.25
        steer = 3.5

        target_kmh = 0
        if keys[pygame.K_w]:
            target_kmh = max_kmh
        if keys[pygame.K_s]:
            target_kmh = -10

        # Smooth speed
        diff = target_kmh - self.kmh
        if abs(diff) < 1:
            self.kmh = target_kmh
        else:
            self.kmh += diff * (accel if diff > 0 else brake)

        self.kmh = max(-10, min(self.kmh, max_kmh))
        self.speed = self.kmh / 10.0

        # Steer
        if keys[pygame.K_a]:
            self.x    -= steer
            self.angle = max(self.angle - 3, -25)
        elif keys[pygame.K_d]:
            self.x    += steer
            self.angle = min(self.angle + 3, 25)
        else:
            self.angle *= 0.85

        if self.hit_timer > 0:
            self.hit_timer -= 1

    def draw(self, surf):
        # Body
        body_color = (200, 50, 50) if self.hit_timer > 0 else (220, 30, 30)
        cx, cy = int(self.x), int(self.y)

        # Shadow
        pygame.draw.ellipse(surf, (20,20,20), (cx-18, cy+22, 36, 10))

        # Rear wheel
        pygame.draw.ellipse(surf, (30,30,30), (cx-10, cy+10, 20, 12))
        pygame.draw.ellipse(surf, (60,60,60), (cx-7,  cy+12, 14, 8))

        # Body
        pts = [(cx-10, cy+14), (cx+10, cy+14),
               (cx+8,  cy-10), (cx-8,  cy-10)]
        pygame.draw.polygon(surf, body_color, pts)

        # Tank / fairing
        pygame.draw.ellipse(surf, (180,20,20), (cx-9, cy-14, 18, 16))

        # Front wheel
        pygame.draw.ellipse(surf, (30,30,30), (cx-8, cy-28, 16, 12))
        pygame.draw.ellipse(surf, (60,60,60), (cx-5, cy-26, 10, 8))

        # Rider helmet
        helmet_color = GREEN if self.hit_timer == 0 else RED
        pygame.draw.circle(surf, (60,40,30), (cx, cy-18), 9)   # head
        pygame.draw.circle(surf, helmet_color, (cx, cy-20), 8) # helmet

    def rect(self):
        return pygame.Rect(int(self.x)-10, int(self.y)-20, 20, 34)


# ── HUD drawing ───────────────────────────────────────────────────────────────
def draw_speedometer(surf, kmh, max_kmh, font, font_sm):
    # Gauge background
    gx, gy, gr = 820, 120, 60
    pygame.draw.circle(surf, (25,25,35), (gx, gy), gr)
    pygame.draw.circle(surf, (50,50,70), (gx, gy), gr, 2)

    # Arc
    start_a = math.radians(220)
    end_a   = math.radians(220 - 240 * (kmh / max(max_kmh,1)))
    for i in range(60):
        a = start_a - math.radians(240 * i / 60)
        frac = i / 60
        r = int(255 * min(frac*2, 1))
        g = int(255 * (1 - max(frac*2-1,0)))
        pygame.draw.arc(surf, (r, g, 0),
                        (gx-gr+8, gy-gr+8, (gr-8)*2, (gr-8)*2),
                        a - math.radians(4), a, 4)

    # Needle
    needle_a = start_a - math.radians(240 * (kmh / max(max_kmh,1)))
    nx = gx + int((gr-14) * math.cos(needle_a))
    ny = gy - int((gr-14) * math.sin(needle_a))
    pygame.draw.line(surf, WHITE, (gx, gy), (nx, ny), 2)
    pygame.draw.circle(surf, WHITE, (gx, gy), 5)

    # Speed text
    spd = font.render(f"{int(kmh)}", True, WHITE)
    surf.blit(spd, (gx - spd.get_width()//2, gy - spd.get_height()//2 + 4))
    unit = font_sm.render("km/h", True, (150,150,180))
    surf.blit(unit, (gx - unit.get_width()//2, gy + 28))


def draw_helmet_status(surf, helmet_on, label, font_sm, flash):
    bx, by = 720, 200
    color  = GREEN if helmet_on else RED
    if not helmet_on and flash:
        color = ORANGE
    pygame.draw.rect(surf, (20,20,30), (bx-60, by-20, 120, 44), border_radius=8)
    pygame.draw.rect(surf, color,      (bx-60, by-20, 120, 44), border_radius=8, width=2)
    icon  = "🪖 HELMET ON" if helmet_on else "⚠ NO HELMET"
    t     = font_sm.render(icon if helmet_on else "NO HELMET", True, color)
    surf.blit(t, (bx - t.get_width()//2, by - t.get_height()//2 + 2))
    sub = font_sm.render(label.split("  ")[-1] if "  " in label else "", True, (120,120,140))
    surf.blit(sub, (bx - sub.get_width()//2, by + 18))


def draw_cam_preview(surf, frame, font_sm, label, helmet_on):
    if frame is not None:
        # Convert BGR→RGB
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        surf_ = pygame.surfarray.make_surface(np.transpose(rgb, (1,0,2)))
        surf.blit(surf_, (W - CAM_W - 10, H - CAM_H - 10))
        border_color = GREEN if helmet_on else RED
        pygame.draw.rect(surf, border_color,
                         (W-CAM_W-10, H-CAM_H-10, CAM_W, CAM_H), 2)
    else:
        pygame.draw.rect(surf, (30,30,40),
                         (W-CAM_W-10, H-CAM_H-10, CAM_W, CAM_H))
        t = font_sm.render("No Camera", True, (100,100,120))
        surf.blit(t, (W-CAM_W+40, H-CAM_H+60))

    lbl = font_sm.render(label, True, GREEN if helmet_on else RED)
    surf.blit(lbl, (W-CAM_W-8, H-CAM_H-22))


def draw_warning_overlay(surf, flash_alpha):
    if flash_alpha <= 0:
        return
    s = pygame.Surface((W, H), pygame.SRCALPHA)
    s.fill((255, 30, 30, min(flash_alpha, 80)))
    surf.blit(s, (0,0))
    # Warning text
    f = pygame.font.SysFont("Impact", 48)
    t = f.render("⚠  WEAR YOUR HELMET!", True, (255,80,80))
    surf.blit(t, (W//2 - t.get_width()//2, H//2 - 30))


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("GRID LOCK  —  Helmet Safety System")
    clock  = pygame.time.Clock()

    font_lg = pygame.font.SysFont("Courier New", 28, bold=True)
    font_md = pygame.font.SysFont("Courier New", 18, bold=True)
    font_sm = pygame.font.SysFont("Courier New", 13)

    # Start detector
    detector = HelmetDetector()

    # Road position (centered)
    road_x = W // 2 - ROAD_W // 2 - 60

    # Stripes
    stripes = [Stripe(y) for y in range(-40, H + 40, 80)]

    # Game state
    bike       = Bike(road_x)
    obstacles  = []
    obs_timer  = 0
    score      = 0
    flash_t    = 0.0
    warn_alpha = 0

    # Keyboard helmet toggle (fallback if no webcam)
    kb_helmet  = True
    no_webcam  = False

    running = True
    state   = "playing"   # playing | dead

    while running:
        dt = clock.tick(FPS)

        # ── Events ────────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_h:
                    kb_helmet = not kb_helmet   # manual toggle
                if event.key == pygame.K_r and state == "dead":
                    bike      = Bike(road_x)
                    obstacles = []
                    score     = 0
                    state     = "playing"

        # ── Detection ─────────────────────────────────────────────────────────
        helmet_on, conf, label, cam_frame = detector.get_state()
        if label == "No webcam":
            no_webcam = True
            helmet_on = kb_helmet
            label     = f"KB: {'HELMET ON' if kb_helmet else 'NO HELMET'}"

        max_kmh = SPEED_HELMET if helmet_on else SPEED_NO_HELMET

        # ── Update ────────────────────────────────────────────────────────────
        if state == "playing":
            keys = pygame.key.get_pressed()
            bike.update(keys, max_kmh, dt)

            # Road boundaries
            if bike.x < road_x + 18:
                bike.x = road_x + 18
            if bike.x > road_x + ROAD_W - 18:
                bike.x = road_x + ROAD_W - 18

            # Stripes
            for s in stripes:
                s.update(bike.speed)

            # Obstacles
            obs_timer -= bike.speed
            if obs_timer <= 0:
                obstacles.append(Obstacle(road_x))
                obs_timer = random.randint(120, 260)

            for obs in obstacles:
                obs.update(bike.speed)
            obstacles = [o for o in obstacles if o.y < H + 60]

            # Collision
            for obs in obstacles:
                if bike.rect().colliderect(obs.rect()):
                    if bike.hit_timer == 0:
                        bike.hit_timer = 40
                        score = max(0, score - 5)
                        obstacles.remove(obs)
                        break

            score += int(bike.speed * 0.05)

            # Warning flash
            if not helmet_on:
                warn_alpha = min(warn_alpha + 3, 80)
                flash_t   += dt * 0.004
            else:
                warn_alpha = max(warn_alpha - 4, 0)
                flash_t    = 0

        # ── Draw ──────────────────────────────────────────────────────────────
        screen.fill(GRASS_COLOR)

        # Kerb strips
        pygame.draw.rect(screen, KERB_COLOR,
                         (road_x - 8, 0, 8, H))
        pygame.draw.rect(screen, KERB_COLOR,
                         (road_x + ROAD_W, 0, 8, H))

        # Road
        pygame.draw.rect(screen, ROAD_COLOR, (road_x, 0, ROAD_W, H))

        # Stripes
        for s in stripes:
            s.draw(screen, road_x)

        # Obstacles
        for obs in obstacles:
            obs.draw(screen)

        # Bike
        if state == "playing":
            bike.draw(screen)

        # Warning overlay
        if not helmet_on:
            draw_warning_overlay(screen, warn_alpha)

        # ── HUD panel ─────────────────────────────────────────────────────────
        pygame.draw.rect(screen, (15,15,22), (680, 0, 220, H))
        pygame.draw.line(screen, (40,40,60), (680, 0), (680, H), 2)

        title = font_md.render("GRID LOCK", True, CYAN)
        screen.blit(title, (790 - title.get_width()//2, 12))

        draw_speedometer(screen, bike.kmh, max_kmh, font_lg, font_sm)
        draw_helmet_status(screen, helmet_on, label, font_sm,
                           int(flash_t * 3) % 2 == 0)

        # Max speed indicator
        lim_color = GREEN if helmet_on else RED
        lim = font_sm.render(f"LIMIT: {max_kmh} km/h", True, lim_color)
        screen.blit(lim, (790 - lim.get_width()//2, 255))

        # Score
        sc = font_md.render(f"SCORE: {score}", True, WHITE)
        screen.blit(sc, (790 - sc.get_width()//2, 290))

        # Controls hint
        for i, txt in enumerate(["W = Accelerate", "S = Brake",
                                  "A/D = Steer",
                                  "H = Toggle helmet (KB)",
                                  "R = Restart  ESC = Quit"]):
            t = font_sm.render(txt, True, (80,80,100))
            screen.blit(t, (688, 320 + i * 18))

        # Webcam preview
        draw_cam_preview(screen, cam_frame, font_sm, label, helmet_on)

        # No webcam notice
        if no_webcam:
            nw = font_sm.render("No webcam — press H to toggle", True, ORANGE)
            screen.blit(nw, (10, 10))

        # Dead screen
        if state == "dead":
            s = pygame.Surface((W, H), pygame.SRCALPHA)
            s.fill((0,0,0,160))
            screen.blit(s, (0,0))
            go = font_lg.render("GAME OVER", True, RED)
            rs = font_md.render("Press R to restart", True, WHITE)
            screen.blit(go, (W//2 - go.get_width()//2, H//2 - 40))
            screen.blit(rs, (W//2 - rs.get_width()//2, H//2 + 10))

        pygame.display.flip()

    detector.stop()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
// Binary Neutron Star: Causal Repair
// Paste this complete file into the Image tab at https://www.shadertoy.com/new
//
// Physics spine: leading adiabatic quadrupole inspiral in G=c=M_total=1,
// equilibrium Love-number tides, and a phenomenological merger/ringdown bridge.
// The visible wave displacement and causal-repair layer are deliberately
// amplified.  See SHADERTOY_GUIDE.md for the exact qualification boundary.

#define CR_QUALITY 1
#define SHOW_WAVE_FIELD 1
#define SHOW_REPAIR_FIELD 1
#define SHOW_DIAGNOSTIC 1
#define SHOW_HORIZON_WIREFRAME 1
#define PROMPT_COLLAPSE_BRANCH 1
#define HAS_MOTS_BAKE 1

const float PI = 3.141592653589793;
const float M1 = 0.5;
const float M2 = 0.5;
const float ETA = 0.25;
const float R1 = 2.94117647;
const float R2 = 2.94117647;
const float R0 = 13.5;
const float RC = 5.88235294;
const float INSPIRAL_END = 0.68;
const float MERGER_END = 0.8;
const float LOOP_SECONDS = 32.0;
const float TOTAL_MASS_SOLAR = 2.7;
const float F2_KHZ = 3.0;
const float POST_DURATION_MS = 4.0;
const float F2_DAMPING_MS = 2.5;
const float F2_OMEGA_M = 0.250676956;
const float OMEGA_C = 0.0700927956;
const float PHASE_C = 73.2133621;
const float INSPIRAL_TIME = 2501.38741;
const float TIDAL1 = 0.016875;
const float TIDAL2 = 0.016875;
const float WAVE_GAIN = 1.0;
const float LENS_GAIN = 0.82;
const float REPAIR_GAIN = 0.22;
const float HORIZON_FORMATION_U = 0.805;

const int MOTS_FRAME_COUNT = 3;
const int MOTS_MODE_COUNT = 4;
const float MOTS_TIMES[3] = float[3](0.82, 0.89, 0.96);
const vec3 MOTS_CENTERS[3] = vec3[3](vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0));
const ivec2 MOTS_MODES[4] = ivec2[4](ivec2(0, 0), ivec2(2, 0), ivec2(4, 0), ivec2(6, 0));
const float MOTS_NORMALIZATION[4] =
    float[4](1.0, 0.630783131, 0.846284375, 1.01710724);
const float MOTS_SHAPE[3 * 4] =
    float[3 * 4](-0.923960514, 0.813619567, 0.0185547619, -0.0203952432, -0.923960514, 0.813619567, 0.0185547619, -0.0203952432, -0.898060182, 0.780442117, 0.017314976, -0.0183961027);
const float MOTS_RESPONSE[3 * 4] =
    float[3 * 4](0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
const float MOTS_LAMBDA0[3] =
    float[3](0.0838468333, 0.081368912, 0.0639441603);
const float MOTS_DISPLAY_SCALE = 6.0;

#if CR_QUALITY == 0
const int MAX_STEPS = 72;
const float HIT_EPSILON = 0.012;
#elif CR_QUALITY == 2
const int MAX_STEPS = 144;
const float HIT_EPSILON = 0.004;
#else
const int MAX_STEPS = 104;
const float HIT_EPSILON = 0.007;
#endif

struct Orbit {
    float u;
    float separation;
    float phase;
    float omega;
    float merge;
    float postMs;
    float ring;
    float remnantPhase;
    float wavePhase22;
    vec3 p1;
    vec3 p2;
};

struct SceneSample {
    float d;
    float material;
    float heat;
};

struct TraceResult {
    float hit;
    float travel;
    float material;
    float heat;
    float waveGlow;
    float repairGlow;
    float repairOutGlow;
    float repairInGlow;
    vec3 p;
    vec3 rd;
};

float horizonBlend(Orbit o) {
#if PROMPT_COLLAPSE_BRANCH
    return smoothstep(HORIZON_FORMATION_U, HORIZON_FORMATION_U + 0.035, o.u);
#else
    return 0.0;
#endif
}

float associatedLegendre(int ell, int order, float x) {
    float pmm = 1.0;
    if (order > 0) {
        float root = sqrt(max(1.0 - x * x, 0.0));
        float factor = 1.0;
        for (int i = 1; i <= 16; ++i) {
            if (i <= order) {
                pmm *= -factor * root;
                factor += 2.0;
            }
        }
    }
    if (ell == order) return pmm;
    float previous = pmm;
    float current = (2.0 * float(order) + 1.0) * x * pmm;
    if (ell == order + 1) return current;
    for (int degree = 2; degree <= 16; ++degree) {
        if (degree >= order + 2 && degree <= ell) {
            float following = (
                (2.0 * float(degree) - 1.0) * x * current
                - float(degree + order - 1) * previous
            ) / float(degree - order);
            previous = current;
            current = following;
        }
    }
    return current;
}

float motsHarmonic(int modeIndex, vec3 n) {
    ivec2 mode = MOTS_MODES[modeIndex];
    int ell = mode.x;
    int m = mode.y;
    if (ell == 0) return 1.0;
    int order = abs(m);
    float polynomial = associatedLegendre(ell, order, clamp(n.z, -1.0, 1.0));
    float azimuth = atan(n.y, n.x);
    if (m == 0) return MOTS_NORMALIZATION[modeIndex] * polynomial;
    float phase = float(order) * azimuth;
    float angular = m > 0 ? cos(phase) : sin(phase);
    return sqrt(2.0) * MOTS_NORMALIZATION[modeIndex] * polynomial * angular;
}

void motsBracket(float u, out int lower, out int upper, out float weight) {
    lower = 0;
    upper = 0;
    weight = 0.0;
    if (u >= MOTS_TIMES[MOTS_FRAME_COUNT - 1]) {
        lower = MOTS_FRAME_COUNT - 1;
        upper = lower;
        return;
    }
    for (int frame = 0; frame < MOTS_FRAME_COUNT - 1; ++frame) {
        if (u >= MOTS_TIMES[frame] && u < MOTS_TIMES[frame + 1]) {
            lower = frame;
            upper = frame + 1;
            weight = (u - MOTS_TIMES[frame])
                / max(MOTS_TIMES[frame + 1] - MOTS_TIMES[frame], 1.0e-6);
        }
    }
}

float motsStability(Orbit o) {
    int lower, upper;
    float weight;
    motsBracket(o.u, lower, upper, weight);
    return mix(MOTS_LAMBDA0[lower], MOTS_LAMBDA0[upper], weight);
}

float motsSurfaceDistance(
    vec3 p,
    Orbit o,
    out vec3 localPoint,
    out float surfaceRadius,
    out float expansionResponse
) {
    int lower, upper;
    float weight;
    motsBracket(o.u, lower, upper, weight);
    vec3 center = MOTS_DISPLAY_SCALE
        * mix(MOTS_CENTERS[lower], MOTS_CENTERS[upper], weight);
    localPoint = p - center;
    vec3 direction = normalize(localPoint + vec3(0.0, 0.0, 1.0e-12));
    float logRadius = 0.0;
    expansionResponse = 0.0;
    for (int modeIndex = 0; modeIndex < MOTS_MODE_COUNT; ++modeIndex) {
        float basis = motsHarmonic(modeIndex, direction);
        int lowerIndex = lower * MOTS_MODE_COUNT + modeIndex;
        int upperIndex = upper * MOTS_MODE_COUNT + modeIndex;
        logRadius += mix(MOTS_SHAPE[lowerIndex], MOTS_SHAPE[upperIndex], weight) * basis;
        expansionResponse += mix(
            MOTS_RESPONSE[lowerIndex], MOTS_RESPONSE[upperIndex], weight
        ) * basis;
    }
    surfaceRadius = MOTS_DISPLAY_SCALE * exp(logRadius);
    return length(localPoint) - surfaceRadius;
}

float saturate(float x) { return clamp(x, 0.0, 1.0); }
vec3 saturate(vec3 x) { return clamp(x, 0.0, 1.0); }

float hash11(float p) {
    p = fract(p * 0.1031);
    p *= p + 33.33;
    p *= p + p;
    return fract(p);
}

float hash31(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.yzx + 33.33);
    return fract((p.x + p.y) * p.z);
}

float valueNoise(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float n000 = hash31(i + vec3(0, 0, 0));
    float n100 = hash31(i + vec3(1, 0, 0));
    float n010 = hash31(i + vec3(0, 1, 0));
    float n110 = hash31(i + vec3(1, 1, 0));
    float n001 = hash31(i + vec3(0, 0, 1));
    float n101 = hash31(i + vec3(1, 0, 1));
    float n011 = hash31(i + vec3(0, 1, 1));
    float n111 = hash31(i + vec3(1, 1, 1));
    return mix(
        mix(mix(n000, n100, f.x), mix(n010, n110, f.x), f.y),
        mix(mix(n001, n101, f.x), mix(n011, n111, f.x), f.y),
        f.z
    );
}

float fbm(vec3 p) {
    float sum = 0.0;
    float weight = 0.55;
    mat3 turn = mat3(
         0.00,  0.80,  0.60,
        -0.80,  0.36, -0.48,
        -0.60, -0.48,  0.64
    );
    for (int i = 0; i < 4; ++i) {
        sum += weight * valueNoise(p);
        p = turn * p * 2.03 + 9.1;
        weight *= 0.48;
    }
    return sum;
}

mat2 rotate2(float a) {
    float c = cos(a), s = sin(a);
    return mat2(c, -s, s, c);
}

float smoothMinimum(float a, float b, float k) {
    float h = saturate(0.5 + 0.5 * (b - a) / k);
    return mix(b, a, h) - k * h * (1.0 - h);
}

float ellipsoidDistance(vec3 p, vec3 radius) {
    float k0 = length(p / radius);
    float k1 = length(p / (radius * radius));
    return k0 * (k0 - 1.0) / max(k1, 1.0e-5);
}

Orbit orbitAt(float u) {
    Orbit o;
    o.u = saturate(u);
    float s = saturate(o.u / INSPIRAL_END);
    float r4 = mix(pow(R0, 4.0), pow(RC, 4.0), s);
    float rInspiral = pow(max(r4, 1.0e-6), 0.25);
    float phaseInspiral = (pow(R0, 2.5) - pow(rInspiral, 2.5)) / (32.0 * ETA);
    float mergeSpan = MERGER_END - INSPIRAL_END;
    float mergeProgress = saturate((o.u - INSPIRAL_END) / mergeSpan);
    // Preserve angular velocity at contact, then decelerate smoothly to the
    // displayed bar-mode rate. This removes the former 49x clock-rate drop.
    float contactRate = OMEGA_C * INSPIRAL_TIME / INSPIRAL_END;
    // The m=2 pattern completes two wave cycles per matter-bar rotation.
    float ringRate = PI * F2_KHZ * POST_DURATION_MS / (1.0 - MERGER_END);
    float mergerPhase = PHASE_C + mergeSpan * (
        contactRate * mergeProgress
        + 0.5 * (ringRate - contactRate) * mergeProgress * mergeProgress
    );
    float phaseAtMerger = PHASE_C + 0.5 * mergeSpan * (contactRate + ringRate);
    float ringPhase = phaseAtMerger + ringRate * max(o.u - MERGER_END, 0.0);
    o.phase = mix(phaseInspiral, mergerPhase, step(INSPIRAL_END, o.u));
    o.phase = mix(o.phase, ringPhase, step(MERGER_END, o.u));
    o.merge = smoothstep(INSPIRAL_END, MERGER_END, o.u);
    o.separation = mix(rInspiral, 0.08 * RC, o.merge);
    o.omega = mix(pow(rInspiral, -1.5), OMEGA_C, step(INSPIRAL_END, o.u));
    o.postMs = POST_DURATION_MS
        * saturate((o.u - MERGER_END) / (1.0 - MERGER_END));
    // Unity through merger, then the damped post-merger envelope.
    o.ring = exp(-o.postMs / F2_DAMPING_MS);
    // The remnant inherits the same C1-continuous source phase.
    o.remnantPhase = o.phase;
    float postF2Phase = 2.0 * phaseAtMerger + 2.0 * PI * F2_KHZ * o.postMs;
    o.wavePhase22 = mix(2.0 * o.phase, postF2Phase, step(MERGER_END, o.u));
    vec3 radial = vec3(cos(o.phase), sin(o.phase), 0.0);
    o.p1 =  M2 * o.separation * radial;
    o.p2 = -M1 * o.separation * radial;
    return o;
}

vec3 corotating(vec3 p, float phase) {
    p.xy = rotate2(-phase) * p.xy;
    return p;
}

SceneSample mapScene(vec3 p, Orbit o) {
    float tideScale = pow(RC / max(o.separation, RC), 3.0);
    // The Love-number displacement is amplified sevenfold so it survives a
    // typical ShaderToy viewport; its phase and separation dependence remain.
    float e1 = min(0.22, 7.0 * TIDAL1 * tideScale);
    float e2 = min(0.22, 7.0 * TIDAL2 * tideScale);
    vec3 q1 = corotating(p - o.p1, o.phase);
    vec3 q2 = corotating(p - o.p2, o.phase);
    float d1 = ellipsoidDistance(q1, R1 * vec3(1.0 + e1, 1.0 - 0.5 * e1, 1.0 - 0.5 * e1));
    float d2 = ellipsoidDistance(q2, R2 * vec3(1.0 + e2, 1.0 - 0.5 * e2, 1.0 - 0.5 * e2));
    float bridge = 0.24 + 0.30 * o.merge;
    float binary = smoothMinimum(d1, d2, bridge);

    float remnantRadius = 0.88 * pow(pow(R1, 3.0) + pow(R2, 3.0), 1.0 / 3.0);
    float bar = 0.17 * o.ring;
    vec3 qr = corotating(p, o.remnantPhase);
    float remnant = ellipsoidDistance(
        qr,
        remnantRadius * vec3(1.0 + bar, 1.0 - bar, 0.88)
    );
    float collapse = horizonBlend(o);
    vec3 horizonRadii = remnantRadius * vec3(0.76, 0.76, 0.65);
    float horizon = ellipsoidDistance(qr, horizonRadii);
#if HAS_MOTS_BAKE
    vec3 bakedLocal;
    float bakedRadius;
    float bakedResponse;
    horizon = motsSurfaceDistance(
        p, o, bakedLocal, bakedRadius, bakedResponse
    );
#endif
    remnant = mix(remnant, horizon, collapse);
    float distanceField = mix(binary, remnant, o.merge);
    float material = mix(d1 < d2 ? 1.0 : 2.0, 3.0, o.merge);
    material = mix(material, 4.0, collapse);
    float neck = exp(-3.5 * length(p.xy)) * exp(-0.8 * abs(p.z));
    float heat = saturate(
        1.7 * o.merge + 0.45 * neck - 0.7 * o.postMs / POST_DURATION_MS
    );
    heat *= 1.0 - collapse;
    return SceneSample(distanceField, material, heat);
}

vec3 sceneNormal(vec3 p, Orbit o) {
    vec2 e = vec2(HIT_EPSILON, 0.0);
    return normalize(vec3(
        mapScene(p + e.xyy, o).d - mapScene(p - e.xyy, o).d,
        mapScene(p + e.yxy, o).d - mapScene(p - e.yxy, o).d,
        mapScene(p + e.yyx, o).d - mapScene(p - e.yyx, o).d
    ));
}

vec3 lensAcceleration(vec3 p, vec3 rd, Orbit o) {
    vec3 d1 = o.p1 - p;
    vec3 d2 = o.p2 - p;
    float s1 = dot(d1, d1) + 0.48 * R1 * R1;
    float s2 = dot(d2, d2) + 0.48 * R2 * R2;
    vec3 g = M1 * d1 / pow(s1, 1.5) + M2 * d2 / pow(s2, 1.5);
    return g - rd * dot(g, rd);
}

vec3 polarizationDirection(vec3 n) {
    vec3 tangent = normalize(vec3(-n.y, n.x, 0.001));
    vec3 bitangent = normalize(cross(n, tangent));
    return normalize(tangent + 0.35 * bitangent);
}

vec3 waveSignals(vec3 p, Orbit o) {
    float radius = max(length(p), RC);
    vec3 n = p / radius;
    float azimuth = atan(n.y, n.x);
    float angular = 0.18 + 0.82 * (1.0 - n.z * n.z);
    float amplitude = 4.0 * ETA / (radius * max(o.separation, 0.45 * RC));
    amplitude *= mix(1.0, o.ring, o.merge);
    float physicalPhase = o.wavePhase22 - 2.0 * o.omega * radius + 2.0 * azimuth;
    float metricWave = amplitude * angular * cos(physicalPhase);
    // Compress wavelength on the visible ribbon only. At literal scale a
    // camera-sized scene contains less than one useful cycle early on.
    float visualPhase = o.wavePhase22 - 12.0 * o.omega * radius + 2.0 * azimuth;
    float waveRibbon = pow(0.5 + 0.5 * cos(visualPhase), 38.0)
        * angular / sqrt(radius);

    // A massive dispersive auxiliary layer, visually distinct from the metric
    // wave.  This is an illustrative Theory-3.3 motif until campaign fitted.
    float repairPhase = o.wavePhase22 - 12.0 * o.omega * radius / 0.78 - azimuth;
    float repairEnvelope = exp(-0.035 * radius) * (0.35 + 0.65 * o.merge);
    float repairRibbon = pow(0.5 + 0.5 * cos(repairPhase), 44.0)
        * repairEnvelope * (0.3 + 0.7 * abs(n.z));
    return vec3(metricWave, waveRibbon, repairRibbon);
}

TraceResult traceScene(vec3 ro, vec3 rd, Orbit o) {
    TraceResult result;
    result.hit = 0.0;
    result.travel = 0.0;
    result.material = 0.0;
    result.heat = 0.0;
    result.waveGlow = 0.0;
    result.repairGlow = 0.0;
    result.repairOutGlow = 0.0;
    result.repairInGlow = 0.0;
    result.p = ro;
    result.rd = rd;
    vec3 p = ro;
    vec3 direction = rd;

    for (int i = 0; i < MAX_STEPS; ++i) {
        SceneSample scene = mapScene(p, o);
        if (scene.d < HIT_EPSILON) {
            result.hit = 1.0;
            result.material = scene.material;
            result.heat = scene.heat;
            break;
        }
        if (result.travel > 58.0) break;
        float stepLength = clamp(scene.d * 0.62, 0.035, 0.58);
        vec3 signals = waveSignals(p, o);
#if SHOW_WAVE_FIELD
        result.waveGlow += stepLength * signals.y;
        direction = normalize(
            direction
            + 0.024 * WAVE_GAIN * signals.x * polarizationDirection(normalize(p)) * stepLength
        );
#endif
#if SHOW_REPAIR_FIELD
        result.repairGlow += stepLength * REPAIR_GAIN * signals.z;
#if HAS_MOTS_BAKE
        vec3 responseLocal;
        float responseRadius;
        float expansionResponse;
        motsSurfaceDistance(
            p, o, responseLocal, responseRadius, expansionResponse
        );
        float responseWeight = clamp(expansionResponse, -1.0, 1.0);
        result.repairOutGlow += stepLength * REPAIR_GAIN * signals.z
            * max(responseWeight, 0.0);
        result.repairInGlow += stepLength * REPAIR_GAIN * signals.z
            * max(-responseWeight, 0.0);
#endif
#endif
        direction = normalize(
            direction + 0.105 * LENS_GAIN * lensAcceleration(p, direction, o) * stepLength
        );
        p += direction * stepLength;
        result.travel += stepLength;
    }
    result.p = p;
    result.rd = direction;
    return result;
}

vec3 blackbodyPalette(float heat) {
    vec3 cold = vec3(0.34, 0.57, 1.00);
    vec3 whiteHot = vec3(1.15, 0.91, 0.72);
    vec3 fusion = vec3(1.55, 0.35, 0.08);
    return mix(mix(cold, whiteHot, saturate(heat * 1.4)), fusion, saturate(heat - 0.55));
}

float horizonWireframe(vec3 localPoint) {
    vec3 n = normalize(localPoint / vec3(0.76, 0.76, 0.65));
    float theta = acos(clamp(n.z, -1.0, 1.0));
    float phi = atan(n.y, n.x);
    float longitude = abs(sin(8.0 * phi));
    float latitude = abs(sin(7.0 * theta));
    float gridDistance = min(longitude, latitude);
    return 1.0 - smoothstep(0.025, 0.115, gridDistance);
}

vec3 shadeSurface(TraceResult hit, vec3 ro, Orbit o) {
    vec3 normal = sceneNormal(hit.p, o);
    vec3 view = normalize(ro - hit.p);
    vec3 lightA = normalize(vec3(-0.45, -0.72, 0.83));
    vec3 lightB = normalize(vec3(0.66, 0.20, -0.40));
    float diffuse = 0.18 + 0.78 * max(dot(normal, lightA), 0.0);
    diffuse += 0.20 * max(dot(normal, lightB), 0.0);
    float rim = pow(1.0 - max(dot(normal, view), 0.0), 3.0);
    vec3 local1 = corotating(hit.p - o.p1, o.phase);
    vec3 local2 = corotating(hit.p - o.p2, o.phase);
    vec3 localStar = length(hit.p - o.p1) < length(hit.p - o.p2) ? local1 : local2;
    vec3 localRemnant = corotating(hit.p, o.remnantPhase);
    // Advect the visible surface pattern with the same continuous orientation
    // as the matter shape, rather than leaving it pinned to world coordinates.
    vec3 texturePoint = mix(localStar, localRemnant, o.merge);
    float granulation = fbm(texturePoint * 2.4 + 7.0 * normal);
    float magneticBand = 0.5 + 0.5 * sin(13.0 * normal.z + 2.0 * granulation);
    vec3 base = hit.material < 1.5
        ? vec3(0.44, 0.66, 1.05)
        : vec3(0.70, 0.50, 1.02);
    base = mix(base, blackbodyPalette(hit.heat), saturate(hit.heat + 0.25 * magneticBand));
    base *= diffuse * (0.80 + 0.28 * granulation);
    base += rim * mix(vec3(0.22, 0.56, 1.0), vec3(1.0, 0.27, 0.08), hit.heat);
    float specular = pow(max(dot(reflect(-lightA, normal), view), 0.0), 28.0);
    base += 0.38 * specular * vec3(0.75, 0.90, 1.0);
    float collapse = horizonBlend(o);
    base = mix(base, vec3(0.0015, 0.003, 0.008) + 0.05 * rim, collapse);
#if SHOW_HORIZON_WIREFRAME
    vec3 wirePoint = localRemnant;
    float expansionResponse = 0.0;
#if HAS_MOTS_BAKE
    float bakedRadius;
    motsSurfaceDistance(
        hit.p, o, wirePoint, bakedRadius, expansionResponse
    );
#endif
    float wire = horizonWireframe(wirePoint);
    float lambda0 = motsStability(o);
    float stabilityColor = clamp(0.5 + 8.0 * lambda0, 0.0, 1.0);
    vec3 wireColor = mix(vec3(0.90, 0.06, 1.18), vec3(0.04, 0.94, 1.28), stabilityColor);
    // Gold marks the physical-normal principal mode approaching the fold.
    float nearZero = exp(-40.0 * abs(lambda0));
    wireColor = mix(wireColor, vec3(1.35, 0.72, 0.08), nearZero);
    wireColor = mix(wireColor, vec3(1.10, 0.34, 0.08), 0.35 * rim);
    base += collapse * wire * wireColor * (0.55 + 0.75 * rim);
    base += collapse * 0.18 * rim * vec3(0.20, 0.62, 1.0);
#endif
    return base;
}

vec3 background(vec3 rd) {
    vec3 direction = normalize(rd);
    float galactic = pow(saturate(1.0 - abs(direction.z + 0.16 * sin(2.0 * direction.x))), 8.0);
    float cloud = fbm(direction * 4.2 + vec3(2.0, -1.0, 7.0));
    vec3 color = vec3(0.002, 0.004, 0.012);
    color += galactic * cloud * vec3(0.055, 0.025, 0.11);
    vec3 cell = floor(direction * 780.0);
    float starSeed = hash31(cell);
    float star = smoothstep(0.9964, 1.0, starSeed);
    float temperature = hash31(cell + 17.0);
    vec3 starColor = mix(vec3(1.0, 0.55, 0.30), vec3(0.52, 0.74, 1.0), temperature);
    color += star * starColor * (1.5 + 5.0 * pow(starSeed, 24.0));
    return color;
}

float reducedWaveform(float u) {
    Orbit o = orbitAt(u);
    float amplitude = RC / max(o.separation, 0.45 * RC);
    amplitude *= mix(1.0, o.ring, o.merge);
    return amplitude * cos(o.wavePhase22);
}

vec3 diagnosticOverlay(vec3 color, vec2 fragCoord, float now) {
#if SHOW_DIAGNOSTIC
    vec2 uv = fragCoord / iResolution.xy;
    vec2 lo = vec2(0.055, 0.065);
    vec2 hi = vec2(0.455, 0.205);
    vec2 q = (uv - lo) / (hi - lo);
    float inside = step(0.0, q.x) * step(q.x, 1.0) * step(0.0, q.y) * step(q.y, 1.0);
    float curve = 0.5 + 0.36 * reducedWaveform(saturate(q.x));
    float line = 1.0 - smoothstep(0.008, 0.023, abs(q.y - curve));
    float axis = 1.0 - smoothstep(0.002, 0.009, abs(q.y - 0.5));
    float cursor = 1.0 - smoothstep(0.002, 0.008, abs(q.x - now));
    vec3 panel = mix(color, vec3(0.004, 0.008, 0.018), 0.68 * inside);
    panel += inside * axis * vec3(0.04, 0.08, 0.13);
    panel += inside * line * vec3(0.28, 0.72, 1.20);
    panel += inside * cursor * vec3(1.20, 0.28, 0.10);
    return panel;
#else
    return color;
#endif
}

vec3 acesToneMap(vec3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

void cameraRay(vec2 fragCoord, float u, out vec3 ro, out vec3 rd) {
    vec2 pixel = (2.0 * fragCoord - iResolution.xy) / iResolution.y;
    float yaw = 0.28 + 0.16 * iTime / LOOP_SECONDS;
    float pitch = 0.24 + 0.025 * sin(0.19 * iTime);
    if (iMouse.z > 0.0) {
        yaw = 2.0 * PI * iMouse.x / iResolution.x;
        pitch = mix(-0.05, 0.82, iMouse.y / iResolution.y);
    }
    float cameraRadius = mix(30.0, 24.0, smoothstep(0.0, MERGER_END, u));
    ro = cameraRadius * vec3(cos(pitch) * cos(yaw), cos(pitch) * sin(yaw), sin(pitch));
    vec3 target = vec3(0.0, 0.0, 0.15);
    vec3 forward = normalize(target - ro);
    vec3 right = normalize(cross(forward, vec3(0.0, 0.0, 1.0)));
    vec3 up = cross(right, forward);
    rd = normalize(forward * 1.72 + pixel.x * right + pixel.y * up);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    float u = fract(iTime / LOOP_SECONDS);
    Orbit orbit = orbitAt(u);
    vec3 ro, rd;
    cameraRay(fragCoord, u, ro, rd);
    TraceResult trace = traceScene(ro, rd, orbit);
    vec3 color = background(trace.rd);
    float wave = 1.0 - exp(-0.34 * WAVE_GAIN * trace.waveGlow);
    float repair = 1.0 - exp(-0.31 * trace.repairGlow);
    float repairOut = 1.0 - exp(-0.40 * trace.repairOutGlow);
    float repairIn = 1.0 - exp(-0.40 * trace.repairInGlow);
    color += wave * vec3(0.045, 0.33, 0.92);
#if HAS_MOTS_BAKE
    color += 0.22 * repair * vec3(0.38, 0.025, 0.68);
    color += repairOut * vec3(0.02, 0.62, 1.10);
    color += repairIn * vec3(0.72, 0.03, 0.42);
#else
    color += repair * vec3(0.38, 0.025, 0.68);
#endif
    if (trace.hit > 0.5) color = shadeSurface(trace, ro, orbit);

    // A fade makes the intentionally looping source history a clean camera cut.
    float loopFade = smoothstep(0.0, 0.018, u) * (1.0 - smoothstep(0.975, 1.0, u));
    color *= loopFade;
    color = diagnosticOverlay(color, fragCoord, u);
    color = acesToneMap(color);
    color = pow(color, vec3(1.0 / 2.2));
    float vignette = 1.0 - 0.18 * dot(
        (fragCoord - 0.5 * iResolution.xy) / iResolution.y,
        (fragCoord - 0.5 * iResolution.xy) / iResolution.y
    );
    fragColor = vec4(color * vignette, 1.0);
}

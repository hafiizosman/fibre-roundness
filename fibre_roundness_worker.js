// fibre_roundness_worker.js
let cvReady = false;
function send(type, data) { self.postMessage(Object.assign({ type }, data || {})); }
send('progress', { pct: 0, msg: 'worker start' });

self.Module = {
  onRuntimeInitialized() {
    cvReady = true;
    send('progress', { pct: 5, msg: 'OpenCV ready' });
  }
};

function tryImport(url) {
  try { importScripts(url); return true; }
  catch (e) { send('progress', { pct: 1, msg: 'import failed ' + url }); return false; }
}
const ok = tryImport('https://docs.opencv.org/4.x/opencv.js') ||
           tryImport('https://cdn.jsdelivr.net/npm/opencv.js@4.10.0/opencv.js');
if (!ok) send('error', { message: 'Failed to load OpenCV.js.' });

function waitForCv(timeout = 15000) {
  return new Promise((res, rej) => {
    const start = Date.now();
    const t = setInterval(() => {
      if (cvReady && typeof cv !== 'undefined' && cv.Mat) { clearInterval(t); res(); }
      else if (Date.now() - start > timeout) { clearInterval(t); rej(new Error('OpenCV init timeout')); }
    }, 15);
  });
}
function toOdd(n) { return n % 2 === 0 ? n + 1 : n; }
function circularity(area, perim) { return (4.0 * Math.PI * area) / ((perim * perim) + 1e-6); }

self.onmessage = async (e) => {
  if ((e.data || {}).type !== 'process') return;
  const {
    width, height, data,
    thresholdRoundness, maxSide, method,
    innerScale, ringDeltaFactor, minCircularity,
    requireIntensityTest, relaxShapeFallback, strongDenoise
  } = e.data;

  try {
    await waitForCv();
    send('progress', { pct: 7, msg: 'Decoding image' });

    const bytes = new Uint8ClampedArray(data);
    const imageData = new ImageData(bytes, width, height);
    let srcFull = cv.matFromImageData(imageData);

    const oh = srcFull.rows, ow = srcFull.cols;
    let pw = ow, ph = oh;
    if (Math.max(ow, oh) > maxSide) {
      const s = maxSide / Math.max(ow, oh);
      pw = Math.max(1, Math.round(ow * s));
      ph = Math.max(1, Math.round(oh * s));
    }
    const scaleX = ow / pw, scaleY = oh / ph;

    let src = new cv.Mat();
    if (pw !== ow || ph !== oh) {
      send('progress', { pct: 10, msg: 'Resizing' });
      cv.resize(srcFull, src, new cv.Size(pw, ph), 0, 0, cv.INTER_AREA);
    } else {
      src = srcFull.clone();
    }

    send('progress', { pct: 15, msg: 'Grayscale' });
    let gray = new cv.Mat();
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0);

    let clahe = new cv.CLAHE(2.0, new cv.Size(8, 8));
    clahe.apply(gray, gray);
    clahe.delete();

    send('progress', { pct: 22, msg: 'Blur' });
    cv.GaussianBlur(gray, gray, new cv.Size(5, 5), 0, 0, cv.BORDER_DEFAULT);

    send('progress', { pct: 28, msg: 'Threshold' });
    let bw = new cv.Mat();
    if (method === 'adaptive') {
      const blk = toOdd(Math.max(9, Math.round(Math.min(pw, ph) / 30)));
      cv.adaptiveThreshold(gray, bw, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, blk, 2);
    } else {
      cv.threshold(gray, bw, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU);
      const m = cv.mean(bw)[0];
      if (m < 127) cv.bitwise_not(bw, bw);
    }

    send('progress', { pct: 34, msg: 'Denoise' });
    let kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(3, 3));
    cv.morphologyEx(bw, bw, cv.MORPH_OPEN, kernel);
    if (strongDenoise) cv.morphologyEx(bw, bw, cv.MORPH_CLOSE, kernel);
    kernel.delete();

    const maskClone = bw.clone();

    send('progress', { pct: 45, msg: 'Contours' });
    let contours = new cv.MatVector();
    let hierarchy = new cv.Mat();
    cv.findContours(bw, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

    const imgArea = src.rows * src.cols;
    const minArea = Math.max(60, Math.floor(imgArea * 0.00002));
    const maxArea = Math.floor(imgArea * 0.18);
    const meanGray = cv.mean(gray)[0];
    const ringDelta = Math.max(5, Math.min(40, Math.round(meanGray * ringDeltaFactor)));

    let below = 0, total = 0;
    const detections = [];

    let maskOuter = new cv.Mat.zeros(src.rows, src.cols, cv.CV_8UC1);
    let maskInner = new cv.Mat.zeros(src.rows, src.cols, cv.CV_8UC1);
    let maskRing = new cv.Mat.zeros(src.rows, src.cols, cv.CV_8UC1);

    const N = contours.size();
    for (let i = 0; i < N; i++) {
      let cnt = contours.get(i);
      const area = cv.contourArea(cnt, false);
      if (area < minArea || area > maxArea || cnt.rows < 5) { cnt.delete(); continue; }

      const rect = cv.boundingRect(cnt);
      const touchesBorder = rect.x <= 0 || rect.y <= 0 || (rect.x + rect.width) >= (src.cols - 1) || (rect.y + rect.height) >= (src.rows - 1);
      if (touchesBorder) { cnt.delete(); continue; }

      const perim = cv.arcLength(cnt, true);
      const circ = circularity(area, perim);
      if (circ < minCircularity) { cnt.delete(); continue; }

      let ellipse = cv.fitEllipse(cnt);
      const major = Math.max(ellipse.size.width, ellipse.size.height);
      const minor = Math.min(ellipse.size.width, ellipse.size.height);
      const roundness = minor / major;

      maskOuter.setTo(new cv.Scalar(0));
      maskInner.setTo(new cv.Scalar(0));
      maskRing.setTo(new cv.Scalar(0));
      const center = ellipse.center;
      const axesOuter = new cv.Size(ellipse.size.width / 2, ellipse.size.height / 2);
      const angle = ellipse.angle;

      cv.ellipse(maskOuter, center, axesOuter, angle, 0, 360, new cv.Scalar(255), -1, cv.LINE_AA);
      const scaleFactor = Math.min(1, Math.max(0.2, innerScale));
      const axesInner = new cv.Size(axesOuter.width * scaleFactor, axesOuter.height * scaleFactor);
      cv.ellipse(maskInner, center, axesInner, angle, 0, 360, new cv.Scalar(255), -1, cv.LINE_AA);
      cv.subtract(maskOuter, maskInner, maskRing);

      const innerMean = cv.mean(gray, maskInner)[0];
      const ringMean = cv.mean(gray, maskRing)[0];
      const delta = ringMean - innerMean;
      const passesIntensity = delta >= ringDelta;

      let accept = false;
      if (requireIntensityTest) {
        if (passesIntensity) accept = true;
        else if (relaxShapeFallback && roundness > 0.9 && circ > 0.7) accept = true;
      } else {
        accept = true;
      }

      if (!accept) { cnt.delete(); continue; }

      total += 1;
      if (roundness < thresholdRoundness) below += 1;

      detections.push({
        cx: ellipse.center.x * scaleX,
        cy: ellipse.center.y * scaleY,
        major: major * scaleX,
        minor: minor * scaleY,
        angle: ellipse.angle,
        roundness,
        circ,
        innerMean,
        ringMean,
        delta,
        below: roundness < thresholdRoundness
      });

      if ((i % 10) === 0 || i === N - 1) {
        const p = 45 + Math.round(45 * (i + 1) / Math.max(1, N));
        send('progress', { pct: p, msg: 'Scoring ' + (i + 1) + '/' + N });
      }

      cnt.delete();
    }

    maskOuter.delete(); maskInner.delete(); maskRing.delete();
    srcFull.delete(); src.delete(); gray.delete(); bw.delete(); contours.delete(); hierarchy.delete();

    send('progress', { pct: 95, msg: 'Packaging' });
    self.postMessage({
      type: 'done',
      total, below, thresholdRoundness,
      detections,
      width: ow, height: oh,
      binaryMask: { data: maskClone.data, rows: maskClone.rows, cols: maskClone.cols }
    }, [maskClone.data.buffer]);
    send('progress', { pct: 100, msg: 'Done' });
    maskClone.delete();
  } catch (err) {
    send('error', { message: (err && err.message) ? err.message : String(err) });
  }
};

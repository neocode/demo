const videoElement = document.getElementsByClassName('input_video')[0];
const canvasElement = document.getElementsByClassName('output_canvas')[0];
const canvasCtx = canvasElement.getContext('2d');
const overlayImg = new Image();
overlayImg.src = 'masks/onix_03.png'

// Optimization: Turn off animated spinner after its hiding animation is done.
const spinner = document.querySelector('.loading');
spinner.ontransitionend = () => {
  spinner.style.display = 'none';
};


function onResults(results) {
  // Hide the spinner.
  document.body.classList.add('loaded');
  // console.log(results);

  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  canvasCtx.drawImage(
      results.image, 0, 0, canvasElement.width, canvasElement.height);
  //canvasCtx.drawImage(overlayImg, 10, 10, 35, 30) // x, y of top-left, width, height

  if (results.multiFaceLandmarks) {
    for (const landmarks of results.multiFaceLandmarks) {
      const maskWidth = Math.round((landmarks[454]['x'] - landmarks[234]['x'])*canvasElement.width) + 80;
      const xLeft = Math.round(landmarks[195]['x']*canvasElement.width - maskWidth/2);
      const yLeft = Math.round(landmarks[195]['y']*canvasElement.height) - 140;
      const maskHeight = Math.round((landmarks[152]['y'] - landmarks[10]['y'])*canvasElement.height) + 100;
      // console.log(xLeft);
      canvasCtx.drawImage(overlayImg, xLeft, yLeft, maskWidth, maskHeight) // x, y of top-left, width, height
      //drawConnectors(canvasCtx, landmarks, FACEMESH_TESSELATION,
      //               {color: '#C0C0C070', lineWidth: 1});
      //drawConnectors(canvasCtx, landmarks, FACEMESH_RIGHT_EYE, {color: '#FF3030'});
      //drawConnectors(canvasCtx, landmarks, FACEMESH_RIGHT_EYEBROW, {color: '#FF3030'});
      //drawConnectors(canvasCtx, landmarks, FACEMESH_LEFT_EYE, {color: '#30FF30'});
      //drawConnectors(canvasCtx, landmarks, FACEMESH_LEFT_EYEBROW, {color: '#30FF30'});
      //drawConnectors(canvasCtx, landmarks, FACEMESH_FACE_OVAL, {color: '#E0E0E0'});
      //drawConnectors(canvasCtx, landmarks, FACEMESH_LIPS, {color: '#E0E0E0'});
      //drawRectangle(canvasCtx, landmarks, {color: '#E0E0E0'})
    }
  }
  canvasCtx.restore();
}

const faceMesh = new FaceMesh({locateFile: (file) => {
  return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`;
}});
faceMesh.setOptions({
  maxNumFaces: 5,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5
});
faceMesh.onResults(onResults);

const camera = new Camera(videoElement, {
  onFrame: async () => {
    await faceMesh.send({image: videoElement});
  },
  width: 1024,
  height: 600
});
camera.start();
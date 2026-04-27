/* MIT License

Copyright (c) 2022 P. Huttunen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

-----------------------
This is example code for using Cat-Dog model in broser with tensorflow.js environment. 
-----------------------
*/

// Function for opening selected input file
var openFile = function(file) {
    var input = file.target;

    var reader = new FileReader();
    reader.onload = function(){
      var dataURL = reader.result;
      var output = document.getElementById('output');
      output.src = dataURL;
    };
    reader.readAsDataURL(input.files[0]);    
    setTimeout(myTimeout1, 1000) 
};

// Function to run image classification after file opening 
function myTimeout1() {
    // Preprocess
    let img_tensor = preprocess(output);

    // Prediction
    processModel(img_tensor).then(
    function(result) {
        console.log(result); 
        document.getElementById("result_text").innerHTML = result; }
    );
  }
    
// Function for image classification 
async function processModel(img_tensor){
    console.log("Load model...")

    let model = await tf.loadGraphModel("./flower_model/model.json");

    console.log("Predict...")
    let prediction = model.execute(img_tensor);

    let classes = ['dandelion', 'daisy', 'tulips', 'sunflowers', 'roses'];

    let predictionData = prediction.dataSync();
    let ind = tf.argMax(prediction, 1).dataSync()[0];

    let confidence = predictionData[ind] * 100;
    let result = classes[ind] + " (" + confidence.toFixed(2) + "%)";

    return result;
}
   
       
// Function for image preprocessing before classification 
function preprocess(imgData)
    {
    console.log("Preprocessing...")
    let tensor = tf.browser.fromPixels(imgData, numChannels= 3)
    //resize to 224 x 224
    let resized = tf.image.resizeBilinear(tensor, [224, 224]).toFloat()
    // Normalize the image 
    let offset = tf.scalar(255.0);
       //let normalized = tf.scalar(1.0).sub(resized.div(offset));
       let normalized = resized.div(offset);
       //We add a dimension to get a batch shape 
       let batched = normalized.expandDims(0)
       return batched
       }


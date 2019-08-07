// Загружаем картинку для предикта
var openFile = function(event) {
    var input = event.target;

    var reader = new FileReader();
    reader.onload = function(){
        var dataURL = reader.result;
        var output = document.getElementById("imagePreview");
        output.src = dataURL; 
    };
    reader.readAsDataURL(input.files[0]);
    document.getElementById("buttonPredict").style.display = "inline";
  };


var predictImage= function(event){
    var output = document.getElementById("imagePreview");
    src = output.src;
    fetch('/predict', {  
        method: 'POST',  
            headers: {
                      'Accept': 'application/json',
                      'Content-Type': 'application/json'
                      },
        body: JSON.stringify({"file": src})
      })
      .then(response => response.json())
      .then(function(response){
      		document.getElementById("imagePredict").src = response['file']; 
      });
};


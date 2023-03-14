var pics = document.querySelectorAll("#images_test img");
var prev = document.querySelector("#img_prev");
var next = document.querySelector("#img_next");
var current = 0;

showPic(current);
function showPic(n){
     for(let i=0; i<pics.length; i++){
        pics[i].style.display = "none";
     }
     pics[n].style.display="block";
}

img_next.onclick = nextPic
img_prev.onclick = prevPic

function nextPic(){
    if (current == pics.length -1){
        current=0;
    }
    else{
        current = current + 1;
    }
    console.log(current);
    showPic(current);
}

function prevPic(){
    if (current == 0){
        current=pics.length -1;
    }
    else{
        current = current - 1;
    }
    console.log(current);
        showPic(current);
}
function abrirModal(imagen){
    var cuadro = document.getElementById("modal");

    var imgModal = document.getElementById("imgModal");

    imgModal.src = imagen.src;

    cuadro.style.display = "flex";
}

function cerrarModal(){
    var cuadro = document.getElementById("modal");

    cuadro.style.display = "none";
}
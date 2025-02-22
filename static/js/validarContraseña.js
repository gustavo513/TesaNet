function validarContraseña(){

    var primera_contrasenia = document.getElementById("contraseña").value;

    var segunda_contrasenia = document.getElementById("contraseña_doble").value;

    var cuadro = document.getElementById("mensaje");

    if(primera_contrasenia == segunda_contrasenia){
        cuadro.style.color = "green";
        cuadro.innerHTML = "Las contraseñas coinciden";
        return true;
    }
    else{
        cuadro.style.color = "red";
        cuadro.innerHTML = "Las contraseñas no son iguales";
        return false;
    }

}
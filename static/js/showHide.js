function showHide() {
    var x = document.getElementById("pass");
    var y = document.getElementById("rpass");
    var z = document.getElementById("passstage");
    var a = document.getElementById("eye");
   
    if (x.type === "password") {
        x.type = "text";
        z.innerHTML = " Hide";
        a.className = "fa fa-eye-slash";
        y.type = "text";

    } else {
        x.type = "password";
        z.innerHTML = " Show";
        a.className = "fa fa-eye";
        y.type = "password";
    }
}
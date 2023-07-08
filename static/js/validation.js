function checkEmail(textbox) {
    if (textbox.value === 'aaa@gmail.com') {
        textbox.setCustomValidity('Email Already Exists');
    }
    else {
        textbox.setCustomValidity('');
    }
    return true;
}
function checkUsername(textbox) {
    if (textbox.value === 'aaa') {
        textbox.setCustomValidity('Username Already Exists');
    }
    else {
        textbox.setCustomValidity('');
    }
    return true;
}
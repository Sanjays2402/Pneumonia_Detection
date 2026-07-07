$(document).ready(function () {
    // Init
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();

    // Upload preview
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(650);
            };
            reader.readAsDataURL(input.files[0]);
        }
    }

    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
        $('#result').removeClass('error').text('');
        $('#result').hide();
        readURL(this);
    });

    // Predict
    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);

        $(this).hide();
        $('.loader').show();

        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                $('.loader').hide();
                $('#result').removeClass('error').fadeIn(600);
                if (data && data.label) {
                    $('#result').text('Result: ' + data.label + ' (' + data.confidence + '% confidence)');
                } else {
                    $('#result').text('Result: ' + JSON.stringify(data));
                }
            },
            error: function (xhr) {
                $('.loader').hide();
                $('#btn-predict').show();
                var msg = 'Something went wrong.';
                try { msg = JSON.parse(xhr.responseText).error || msg; } catch (e) { /* ignore */ }
                $('#result').addClass('error').fadeIn(600).text('Error: ' + msg);
            }
        });
    });
});


<?php
$servername = "localhost";
$username = "root";
$password = "";
$dbname = "feedback";

// Create connection
$conn = new mysqli($servername, $username, $password, $dbname);
// Check connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $firstname = $conn->real_escape_string($_POST["NAME"]);
    $projectname = $conn->real_escape_string($_POST["Project"]);
    $unitnumber = $conn->real_escape_string($_POST['UnitNo']);
    $mobilenumber = $conn->real_escape_string($_POST['mobile_number']);
    $Reason_visit = $conn->real_escape_string($_POST['Reason']);
    $review_counter = $conn->real_escape_string($_POST['exr1']);
    $review_visit = $conn->real_escape_string($_POST['exr2']);
    $review_all = $conn->real_escape_string($_POST['exr3']);
    $improvement =  isset($_POST['imp']) ? $conn->real_escape_string($_POST['imp']) : '';


    // Insert the form data into the database
    $sql = "INSERT INTO feed (firstname, Projectname, unitnumber, mobilenumber, reason,executive_counter_rating	, visited_rating, overall_rating, improvement)
    VALUES ('$firstname', '$projectname', '$unitnumber', '$mobilenumber', '$Reason_visit', '$review_counter', '$review_visit', '$review_all', '$improvement')";

    if ($conn->query($sql) === TRUE) {
        echo "New record created successfully. <br> Thank you for your feedback!";
    } else {
        echo "Error: " . $sql . "<br>" . $conn->error;
    }
}

$conn->close();

?>





<?php
echo "happy bday";
$firstName = $_POST['NAME'];
$projectname=$_post['Project']
$unitnumber = $_POST['Unitno'];
$mobilenumber = $_POST['mobile_number'];
$Reason_visit=$_POST['Reason'];
$review_counter = $_POST['exr1'];
$review_visit = $_POST['exr2'];
$review_all = $_POST['exr3'];
$improvement=$_post['imp'];
echo "happy bday";
// conecting details 
$host = 'localhost';
$port = '5555';
$database = 'gaur_feedbackform';
$username = 'postgres';
$password = 'Intern@8000';
echo "happy bday";

// Connect to the PostgreSQL database
$conn = pg_connect("host=$host port=$port dbname=$database user=$username password=$password");
if (!$conn) {
    echo "Failed to connect to the database.";
    exit;
}
echo "happy bday";

// Insert the form data into the database
$sql = "INSERT INTO feedback VALUES ('$firstname', '$projectname', $unitnumber , '$mobilenumber', '$Reason_visit', $review_counter, $review_visit, $review_all, '$improvement')";

if ($conn->query($sql) === TRUE) {
    echo "Thank you for your feedback!";
} else {
    echo "Error: " . $sql . "<br>" . $conn->error;
} 
// close server 
$conn->close();


?>
function refreshDashboard() {
    alert("Dashboard refreshed successfully.");
}


function filterCustomers() {

    const input = document.getElementById("customerSearch");

    const filter = input.value.toLowerCase();

    const table = document.getElementById("customerTable");

    const rows = table
        .getElementsByTagName("tbody")[0]
        .getElementsByTagName("tr");


    for (let i = 0; i < rows.length; i++) {

        const customerName =
            rows[i]
                .getElementsByTagName("td")[1]
                .textContent
                .toLowerCase();


        if (customerName.includes(filter)) {
            rows[i].style.display = "";
        } else {
            rows[i].style.display = "none";
        }
    }
}

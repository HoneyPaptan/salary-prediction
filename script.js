document.addEventListener('DOMContentLoaded', function() {
    const jobTitles = {
        "Software Engineer": 159,
        "Data Analyst": 17,
        "Senior Manager": 130,
        "Sales Associate": 101,
        "Director": 22,
        "Marketing Analyst": 81,
        "Product Manager": 93,
        "Sales Manager": 104,
        "Marketing Coordinator": 82,
        "Senior Scientist": 150,
        "Software Developer": 158,
        "HR Manager": 40,
        "Financial Analyst": 36,
        "Project Manager": 96,
        "Customer Service Rep": 13,
        "Operations Manager": 89,
        "Marketing Manager": 83,
        "Senior Engineer": 116,
        "Data Entry Clerk": 18,
        "Sales Director": 102,
        "Business Analyst": 3,
        "VP of Operations": 172,
        "IT Support": 44,
        "Recruiter": 98,
        "Financial Manager": 37,
        "Social Media Specialist": 157,
        "Software Manager": 160,
        "Junior Developer": 57,
        "Senior Consultant": 112,
        "Product Designer": 92,
        "CEO": 6,
        "Accountant": 1,
        "Data Scientist": 19,
        "Marketing Specialist": 84,
        "VP of Finance": 171,
        "Graphic Designer": 38,
        "UX Designer": 169,
        "Project Engineer": 95,
        "Customer Success Rep": 16,
        "Sales Executive": 103,
        "Business Intelligence Analyst": 5,
        "UX Researcher": 170,
        "Junior Designer": 56,
        "Technical Writer": 167,
        "HR Generalist": 39,
        "Creative Director": 11,
        "Junior Accountant": 47,
        "Help Desk Analyst": 41,
        "Chief Technology Officer": 8,
        "Copywriter": 10,
        "Account Manager": 0,
        "Director of Marketing": 29,
        "Junior Web Developer": 80,
        "Supply Chain Manager": 164,
        "Network Engineer": 85,
        "Administrative Assistant": 2,
        "Strategy Consultant": 162,
        "Junior Account Manager": 46,
        "Senior Financial Analyst": 118,
        "Web Developer": 173,
        "Training Specialist": 168,
        "Research Director": 99,
        "Technical Support Specialist": 166,
        "Public Relations Manager": 97,
        "Junior Software Developer": 76,
        "Operations Analyst": 87,
        "Senior Marketing Manager": 134,
        "Office Manager": 86,
        "Principal Scientist": 91,
        "Junior HR Generalist": 61,
        "Senior Project Manager": 144,
        "Chief Data Officer": 7,
        "Junior Sales Representative": 73,
        "Junior Marketing Analyst": 62,
        "Business Development Manager": 4,
        "Senior Product Manager": 141,
        "Senior Business Analyst": 110,
        "Senior HR Manager": 122,
        "Senior Project Coordinator": 143,
        "Junior Marketing Manager": 64,
        "Junior Operations Analyst": 66,
        "Junior Project Manager": 70,
        "Senior Sales Manager": 148,
        "Junior Web Designer": 79,
        "Senior Training Specialist": 154,
        "Senior Research Scientist": 146,
        "Junior Business Analyst": 49,
        "Junior Recruiter": 71,
        "Senior Business Development Manager": 111,
        "Senior Product Designer": 139,
        "Junior Customer Support Specialist": 53,
        "Senior Marketing Analyst": 131,
        "Senior IT Support Specialist": 44
    };


    const jobTitleSelect = document.getElementById('job_title');

    // Populate the Select2 dropdown with options
    Object.keys(jobTitles).forEach(title => {
        const option = new Option(title, jobTitles[title]);
        jobTitleSelect.appendChild(option);
    });

    // Initialize Select2
    $(jobTitleSelect).select2({
        placeholder: 'Select a job title',
        allowClear: true,
        width: '100%'
    });

    // Form submission handling
    document.getElementById('predictionForm').addEventListener('submit', function(e) {
        e.preventDefault();

        const age = document.getElementById('age').value;
        const gender = document.getElementById('gender').value;
        const degree = document.getElementById('degree').value;
        const job_title = jobTitleSelect.value; // Get the selected job title value
        const experience = document.getElementById('experience').value;

        document.getElementById('loading').style.display = 'block';
        document.getElementById('result').innerText = '';

        fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                Age: age,
                Gender: gender,
                degree: degree,
                Job_Title: job_title,
                Experience: experience
            })
        })
        .then(response => response.json())
        .then(data => {
            document.getElementById('loading').style.display = 'none';
            document.getElementById('result').innerText = 'Predicted Salary: ' + data.salary + '₹';
        })
        .catch(error => {
            document.getElementById('loading').style.display = 'none';
            document.getElementById('result').innerText = 'Error: ' + error;
            console.error('Error:', error);
        });
    });
});

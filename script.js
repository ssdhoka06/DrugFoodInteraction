$(document).ready(function() {
    // Initialize Select2 dropdowns with data from Flask template
    function initializeSelect2() {
        // Drug dropdown
        $('#drug').select2({
            placeholder: "Select a medication...",
            allowClear: true,
            width: '100%',
            data: window.drugs || []  // Use data passed from Flask template
        });

        // Food dropdown
        $('#food').select2({
            placeholder: "Select a food...",
            allowClear: true,
            width: '100%',
            data: window.foods || []  // Use data passed from Flask template
        });
    }

    // Initialize dropdowns when page loads
    initializeSelect2();

    // Handle form submission
    $('#interactionForm').on('submit', function(e) {
        e.preventDefault();
        
        const drug = $('#drug').val();
        const food = $('#food').val();
        
        if (!drug || !food) {
            showAlert('Please select both a medication and a food item', 'error');
            return;
        }
        
        // Show loading state
        showLoading(drug, food);
        
        // Send prediction request to backend
        $.ajax({
            url: '/api/predict',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                drug: drug,
                food: food
            }),
            success: function(response) {
                if (response.success) {
                    displayPredictionResults(response.data);
                } else {
                    showAlert(response.error || 'Error processing your request', 'error');
                }
            },
            error: function(xhr) {
                const errorMsg = xhr.responseJSON?.error || 'Server error occurred';
                showAlert(errorMsg, 'error');
            }
        });
    });

    // Display loading state
    function showLoading(drug, food) {
        $('#output').html(`
            <div class="loading-state">
                <div class="spinner"></div>
                <p>Analyzing interaction between <strong>${drug}</strong> and <strong>${food}</strong>...</p>
            </div>
        `);
    }

    // Show alert message
    function showAlert(message, type) {
        $('#output').html(`
            <div class="alert alert-${type}">
                <i class="fas fa-${type === 'error' ? 'exclamation-triangle' : 'info-circle'}"></i>
                <span>${message}</span>
            </div>
        `);
    }

    // Display prediction results
    function displayPredictionResults(data) {
        const riskClass = data.risk_level.toLowerCase();
        const mechanismText = data.mechanism.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
        
        let mechanismDetails = '';
        if (data.mechanism !== 'unknown') {
            mechanismDetails = `
                <div class="mechanism-details">
                    <h4><i class="fas fa-microscope"></i> Interaction Mechanism</h4>
                    <p>${data.explanation}</p>
                    <div class="mechanism-tag">${mechanismText}</div>
                </div>
            `;
        }
        
        let similarInteractions = '';
        if (data.similar_interactions && data.similar_interactions.length > 0) {
            similarInteractions = `
                <div class="similar-interactions">
                    <h4><i class="fas fa-random"></i> Similar Interactions</h4>
                    <div class="similar-list">
                        ${data.similar_interactions.map(interaction => `
                            <div class="similar-item">
                                <div class="similar-pair">${interaction.drug} + ${interaction.food}</div>
                                <div class="similar-risk ${interaction.risk_level.toLowerCase()}-risk">${interaction.risk_level}</div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            `;
        }
        
        $('#output').html(`
            <div class="prediction-header">
                <div class="drug-food-pair">
                    <span class="drug-name">${data.drug}</span>
                    <span class="interaction-plus">+</span>
                    <span class="food-name">${data.food}</span>
                </div>
                <div class="prediction-confidence">
                    <span class="confidence-tag">${Math.round(data.probability * 100)}% Confidence</span>
                </div>
            </div>
            
            <div class="prediction-summary ${riskClass}-risk">
                <div class="risk-level">
                    <div class="risk-icon">
                        ${data.risk_level === 'HIGH' ? '<i class="fas fa-exclamation-triangle"></i>' : 
                          data.risk_level === 'MODERATE' ? '<i class="fas fa-info-circle"></i>' : 
                          '<i class="fas fa-check-circle"></i>'}
                    </div>
                    <div class="risk-text">
                        <h3>${data.interaction_predicted ? 'Interaction Detected' : 'No Significant Interaction'}</h3>
                        <p class="risk-category ${riskClass}-risk">${data.risk_level} Risk</p>
                    </div>
                </div>
                
                ${mechanismDetails}
                
                <div class="clinical-recommendations">
                    <h4><i class="fas fa-stethoscope"></i> Clinical Recommendations</h4>
                    <p>${data.recommendations}</p>
                </div>
                
                ${similarInteractions}
                
                <div class="prediction-actions">
                    <button class="btn-save-to-profile">
                        <i class="fas fa-save"></i> Save to Profile
                    </button>
                    <button class="btn-generate-report">
                        <i class="fas fa-file-pdf"></i> Generate PDF Report
                    </button>
                </div>
            </div>
        `);
        
        // Add event listeners for action buttons
        $('.btn-save-to-profile').click(function() {
            saveToProfile(data);
        });
        
        $('.btn-generate-report').click(function() {
            generateReport(data);
        });
    }

    // Save to profile function
    function saveToProfile(data) {
        // In a real implementation, this would make an API call to save to user's profile
        alert('This would save the interaction to your profile when logged in');
    }

    // Generate PDF report function
    function generateReport(data) {
        // In a real implementation, this would call your Flask endpoint to generate a PDF
        alert('This would generate a PDF report when implemented');
    }

    // Clear form button
    $('.btn-clear').click(function() {
        $('#drug').val(null).trigger('change');
        $('#food').val(null).trigger('change');
        $('#output').html(`
            <div class="results-placeholder">
                <i class="fas fa-microscope"></i>
                <p>Enter a drug and food combination to see interaction analysis</p>
            </div>
        `);
    });

    // Login/Signup Modal Toggle
    $('.btn-login').click(function() {
        $('#loginModal').addClass('active');
    });
    
    $('.btn-signup').click(function() {
        $('#signupModal').addClass('active');
    });
    
    $('.modal-close, .switch-to-signup').click(function() {
        $('#loginModal').removeClass('active');
        $('#signupModal').addClass('active');
    });
    
    $('.switch-to-login').click(function() {
        $('#signupModal').removeClass('active');
        $('#loginModal').addClass('active');
    });
    
    // Close modal when clicking outside
    $(document).click(function(e) {
        if ($(e.target).hasClass('modal')) {
            $('.modal').removeClass('active');
        }
    });
    
    // Mobile menu toggle
    $('.mobile-menu-btn').click(function() {
        $('.nav-links').toggleClass('active');
    });
    
    // Smooth scrolling for navigation links
    $('a[href^="#"]').on('click', function(e) {
        e.preventDefault();
        
        $('.nav-links').removeClass('active');
        
        const target = $(this).attr('href');
        if (target === '#') return;
        
        $('html, body').animate({
            scrollTop: $(target).offset().top - 80
        }, 500);
    });
    
    // Add active class to nav links based on scroll position
    $(window).scroll(function() {
        const scrollPosition = $(this).scrollTop();
        
        $('section').each(function() {
            const currentSection = $(this);
            const sectionTop = currentSection.offset().top - 100;
            const sectionId = currentSection.attr('id');
            
            if (scrollPosition >= sectionTop && 
                scrollPosition < sectionTop + currentSection.outerHeight()) {
                $('.nav-links a').removeClass('active');
                $('.nav-links a[href="#' + sectionId + '"]').addClass('active');
            }
        });
    });
    
    // Initialize with first section active
    $('.nav-links a[href="#predictor"]').addClass('active');
});

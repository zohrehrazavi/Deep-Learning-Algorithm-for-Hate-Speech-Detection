
  (function ($) {
  
  "use strict";

    // MENU
    $('.navbar-collapse a').on('click',function(){
      $(".navbar-collapse").collapse('hide');
    });
    
    // CUSTOM LINK
    $('.smoothscroll').click(function(){
      var el = $(this).attr('href');
      var elWrapped = $(el);
      var header_height = $('.navbar').height();
  
      scrollToDiv(elWrapped,header_height);
      return false;
  
      function scrollToDiv(element,navheight){
        var offset = element.offset();
        var offsetTop = offset.top;
        var totalScroll = offsetTop-navheight;
  
        $('body,html').animate({
        scrollTop: totalScroll
        }, 300);
      }
    });

    $(window).on('scroll', function(){
      function isScrollIntoView(elem, index) {
        var docViewTop = $(window).scrollTop();
        var docViewBottom = docViewTop + $(window).height();
        var elemTop = $(elem).offset().top;
        var elemBottom = elemTop + $(window).height()*.5;
        if(elemBottom <= docViewBottom && elemTop >= docViewTop) {
          $(elem).addClass('active');
        }
        if(!(elemBottom <= docViewBottom)) {
          $(elem).removeClass('active');
        }
        var MainTimelineContainer = $('#vertical-scrollable-timeline')[0];
        if (MainTimelineContainer) {
          var MainTimelineContainerBottom = MainTimelineContainer.getBoundingClientRect().bottom - $(window).height()*.5;
          $(MainTimelineContainer).find('.inner').css('height',MainTimelineContainerBottom+'px');
        }
      }
      var timeline = $('#vertical-scrollable-timeline li');
      Array.from(timeline).forEach(isScrollIntoView);
    });
  
  })(window.jQuery);

  // Make navbar sticky
  $(document).ready(function() {
    if ($('.navbar').length) {
      $('.navbar').sticky({
        topSpacing: 0,
        zIndex: 999
      });
    }
  });

  // Hate Speech Detection - Analyze Functionality
  document.addEventListener('DOMContentLoaded', function() {
    const analyzeForm = document.getElementById('analyzeForm');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const textInput = document.getElementById('textInput');
    const resultsSection = document.getElementById('results-section');
    
    if (analyzeForm && analyzeBtn && textInput) {
      // Handle form submission
      analyzeForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const inputText = textInput.value.trim();
        
        if (inputText === '') {
          alert('Please enter some text to analyze.');
          return;
        }
        
        // Show loading state
        analyzeBtn.disabled = true;
        const originalText = analyzeBtn.textContent;
        analyzeBtn.textContent = 'Analyzing...';
        
        // Submit form via AJAX to maintain smooth UX
        const formData = new FormData(analyzeForm);
        
        fetch(analyzeForm.action, {
          method: 'POST',
          body: formData
        })
        .then(response => response.text())
        .then(html => {
          // Create a temporary div to parse the response
          const tempDiv = document.createElement('div');
          tempDiv.innerHTML = html;
          
          // Extract results section from response
          const responseResultsSection = tempDiv.querySelector('#results-section');
          
          if (responseResultsSection) {
            // Update or create results section
            if (resultsSection) {
              resultsSection.innerHTML = responseResultsSection.innerHTML;
              resultsSection.style.display = 'block';
            } else {
              // Create results section if it doesn't exist
              const newResultsSection = responseResultsSection.cloneNode(true);
              newResultsSection.style.display = 'block';
              const heroSection = document.getElementById('section_1');
              if (heroSection && heroSection.parentNode) {
                heroSection.parentNode.insertBefore(newResultsSection, heroSection.nextSibling);
              }
            }
            
            // Scroll to results smoothly
            setTimeout(function() {
              const targetSection = document.getElementById('results-section');
              if (targetSection) {
                targetSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
              }
            }, 100);
          }
          
          // Reset button
          analyzeBtn.disabled = false;
          analyzeBtn.textContent = originalText;
        })
        .catch(error => {
          console.error('Error:', error);
          alert('An error occurred while analyzing the text. Please try again.');
          analyzeBtn.disabled = false;
          analyzeBtn.textContent = originalText;
        });
      });
      
      // Allow Enter key to trigger analysis (Ctrl+Enter)
      textInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && e.ctrlKey) {
          e.preventDefault();
          analyzeForm.dispatchEvent(new Event('submit'));
        }
      });
    }
    
    // Scroll to results if they exist on page load
    if (resultsSection && resultsSection.style.display !== 'none') {
      setTimeout(function() {
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 500);
    }
  });


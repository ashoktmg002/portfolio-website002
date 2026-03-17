(function ($) {
    "use strict";
    
    // Back to top button
    $(window).scroll(function () {
        if ($(this).scrollTop() > 200) {
            $('.back-to-top').fadeIn('slow');
        } else {
            $('.back-to-top').fadeOut('slow');
        }
    });
    $('.back-to-top').click(function () {
        $('html, body').animate({scrollTop: 0}, 1500, 'easeInOutExpo');
        return false;
    });
    
    
    // Sticky Navbar
    $(window).scroll(function () {
        if ($(this).scrollTop() > 294) {
            $('.nav-bar').addClass('nav-sticky');
            $('.carousel, .page-header').css("margin-top", "73px");
        } else {
            $('.nav-bar').removeClass('nav-sticky');
            $('.carousel, .page-header').css("margin-top", "0");
        }
    });


    // Dynamic Carousel Height
    function adjustCarouselHeight() {
        var topBar = $('.top-bar');
        var navBar = $('.nav-bar');
        var totalHeight = topBar.outerHeight(true) + navBar.outerHeight(true);
        var carouselHeight = 'calc(100vh - ' + totalHeight + 'px)';

        $('.carousel').css('height', carouselHeight);
        $('.carousel .carousel-caption').css('height', carouselHeight);
    }

    $(window).on('load resize', adjustCarouselHeight);


    // Dropdown on mouse hover + close after selection
    $(document).ready(function () {

        function toggleNavbarMethod() {
            if ($(window).width() > 992) {
                $('.navbar .dropdown').off('mouseover mouseout').on('mouseover', function () {
                    $(this).addClass('show');
                    $(this).find('.dropdown-toggle').attr('aria-expanded', 'true');
                    $(this).find('.dropdown-menu').addClass('show');
                }).on('mouseout', function () {
                    $(this).removeClass('show');
                    $(this).find('.dropdown-toggle').attr('aria-expanded', 'false');
                    $(this).find('.dropdown-menu').removeClass('show');
                });
            } else {
                $('.navbar .dropdown').off('mouseover mouseout');
            }
        }

        toggleNavbarMethod();
        $(window).resize(toggleNavbarMethod);

        // Close dropdown after language selection (desktop + mobile)
        $(document).on('click', '.dropdown-menu .dropdown-item', function () {
            var $dropdownContainer = $(this).closest('.dropdown');
            var $dropdownMenu = $dropdownContainer.find('.dropdown-menu');
            var $toggle = $dropdownContainer.find('.dropdown-toggle');

            $dropdownMenu.removeClass('show').hide();
            $dropdownContainer.removeClass('show');
            $toggle.attr('aria-expanded', 'false');

            setTimeout(function () {
                $dropdownMenu.removeAttr('style');
            }, 300);

            var $navbarCollapse = $('#navbarCollapse');
            if ($navbarCollapse.hasClass('show')) {
                $navbarCollapse.collapse('hide');
            }
        });

        // Collapse navbar when anchor links are clicked on mobile
        $('.navbar-nav a.nav-link').on('click', function () {
            var $navbarCollapse = $('#navbarCollapse');
            if ($navbarCollapse.hasClass('show')) {
                $navbarCollapse.collapse('hide');
            }
        });

    });

    // Portfolio isotope and filter
    var portfolioIsotope = $('.portfolio-container').isotope({
        itemSelector: '.portfolio-item',
        layoutMode: 'fitRows'
    });

    $('#portfolio-flters li').on('click', function () {
        $("#portfolio-flters li").removeClass('filter-active');
        $(this).addClass('filter-active');

        portfolioIsotope.isotope({filter: $(this).data('filter')});
    });
    
})(jQuery);
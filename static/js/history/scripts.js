/*!
* Start Bootstrap - Freelancer v7.0.6 (https://startbootstrap.com/theme/freelancer)
* Copyright 2013-2022 Start Bootstrap
* Licensed under MIT (https://github.com/StartBootstrap/startbootstrap-freelancer/blob/master/LICENSE)
*/
//
// Scripts
// 

window.addEventListener('DOMContentLoaded', event => {

    // Navbar shrink function
    var navbarShrink = function () {
        const navbarCollapsible = document.body.querySelector('#mainNav');
        if (!navbarCollapsible) {
            return;
        }
        if (window.scrollY === 0) {
            navbarCollapsible.classList.remove('navbar-shrink')
        } else {
            navbarCollapsible.classList.add('navbar-shrink')
        }

    };

    // Shrink the navbar 
    navbarShrink();

    // Shrink the navbar when page is scrolled
    document.addEventListener('scroll', navbarShrink);

    // Activate Bootstrap scrollspy on the main nav element
    const mainNav = document.body.querySelector('#mainNav');
    if (mainNav) {
        new bootstrap.ScrollSpy(document.body, {
            target: '#mainNav',
            offset: 72,
        });
    };

    // Collapse responsive navbar when toggler is visible
    const navbarToggler = document.body.querySelector('.navbar-toggler');
    const responsiveNavItems = [].slice.call(
        document.querySelectorAll('#navbarResponsive .nav-link')
    );
    responsiveNavItems.map(function (responsiveNavItem) {
        responsiveNavItem.addEventListener('click', () => {
            if (window.getComputedStyle(navbarToggler).display !== 'none') {
                navbarToggler.click();
            }
        });
    });

});


class LoopingElement {
    constructor(element, currentTranslation, speed) {
      this.element = element
      this.currentTranslation = currentTranslation
      this.speed = speed
      this.direction = true
      this.scrollTop = 0
      this.metric = 100
      
      this.lerp = {
        current: this.currentTranslation,
        target: this.currentTranslation,
        ease: 0.2
      }
      
      this.events()
      this.render()
    }
    
    // events() {
    //   window.addEventListener("scroll", (e) => {
    //     let direction = window.pageYOffset || document.documentElement.scrollTop
    //     if (direction > this.scrollTop) {
    //       this.direction = true
    //       this.lerp.target += this.speed * 5
    //     } else {
    //       this.direction = false
    //       this.lerp.target -= this.speed * 5
    //     }
    //     this.scrollTop = direction <= 0 ? 0 : direction
    //   })
    // }
    
    lerpFunc(current, target, ease) {
      this.lerp.current = current * (1 - ease) + target * ease
    }
    
    right() {
      this.lerp.target += this.speed
      if(this.lerp.target > this.metric) {
        this.lerp.current -= this.metric * 2
        this.lerp.target -= this.metric * 2
      }
    }
    
    animate() {
  //    this.direction ? this.right() : this.left()
      this.right()
      this.lerpFunc(this.lerp.current, this.lerp.target, this.lerp.ease)
      
      this.element.style.transform = `translateX(${this.lerp.current}%)`
    }
    
    render() {
      this.animate()
      window.requestAnimationFrame(() => this.render())
    }
  }
  
  let imagesArray = document.querySelectorAll(".images-wrapper")
  
  new LoopingElement(imagesArray[0], 0, 0.1);
  new LoopingElement(imagesArray[1], -100, 0.1);
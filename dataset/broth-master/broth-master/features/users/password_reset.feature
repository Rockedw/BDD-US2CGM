Feature: Sign in
  In order to get access to protected sections of the site
  A user
  Should be able to reset their password
  

  Scenario: User requests a password reset
    Given a user exists with an email of "joe@sixpack.com"
    When I go to the password reset page
    And I fill in "Email Address" with "joe@sixpack.com"
    And I press "Reset Password"
    Then I should see "You will receive an email with instructions about how to reset your password in a few minutes."
  
  
  
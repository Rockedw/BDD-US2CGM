Feature: Invites

  In order to get access to protected sections of the beta site
  As a visitor
  I should be able to request and use invites
  
  Background:
    Given site settings have been enabled for "beta_invites"
    Given an email template for "invitation" exists
    Given an email template for "welcome" exists
  
  Scenario: Visitor should be notified that an invite is required to signup
    When I go to the sign up page
    Then I should see "At this time an invite is required"
    
  Scenario: Visitor can't signup without an invite
    Given no invites exist
    And a user exists with an email of "joe@sixpack.com"
    When I go to the sign up page
    And I fill in "Email" with "joe@fourbeansoup.com"
    And I fill in "Password" with "passw0rd"
    And I fill in "Password confirmation" with "passw0rd"
    And I press "Sign Up"
    Then I should not see "You have signed up successfully"
    And I should see "Email doesn't have an invite yet"
  
  Scenario: Visitor can signup with an invite
    Given an invite exists for "joe@fourbeansoup.com"
    And a user exists with an email of "joe@sixpack.com"
    When I go to the sign up page
    And I fill in "Email" with "joe@fourbeansoup.com"
    And I fill in "Password" with "passw0rd"
    And I fill in "Password confirmation" with "passw0rd"
    And I press "Sign Up"
    Then I should see "You have signed up successfully"
    
  Scenario: Visitor can request an invite
    Given no invites exist
    When I go to the new invite page
    And I fill in "Email" with "joe@fourbeansoup.com"
    And I press "Request Invite"
    Then I should see "Your invite request has been sent to a site admin"
  
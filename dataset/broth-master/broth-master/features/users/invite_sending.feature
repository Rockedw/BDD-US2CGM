Feature: Send an invite
  In order to have friends use the site
  A user
  Should be able to send invites
  
  Background:
    Given site settings have been enabled for "beta_invites"
    Given an email template for "invitation" exists
  
  Scenario: A user sends an invite
    Given I am signed up and signed in as "email@person.com/password"
    And no invites exist
    When I go to the new invite page
    And I fill in "Email" with "joe@myfriend.com"
    And I press "Send Invite"
    Then I should see "Your invite has been sent"
  
  
  